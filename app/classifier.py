import re
from dataclasses import dataclass
from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import get_settings


STAGES = ("I", "II", "III", "IV")
HYPOTHESES = {
    "I": "This note describes stage one breast cancer.",
    "II": "This note describes stage two breast cancer.",
    "III": "This note describes stage three breast cancer.",
    "IV": "This note describes stage four metastatic breast cancer.",
}


@dataclass(frozen=True)
class Classification:
    label: str
    confidence: float
    probabilities: dict[str, float]
    method: str


@lru_cache
def _load_model():
    model_name = get_settings().classifier_model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def _entailment_index(model) -> int:
    labels = {str(value).lower(): int(key) for key, value in model.config.id2label.items()}
    for name, index in labels.items():
        if "entail" in name:
            return index
    raise RuntimeError("Classifier checkpoint has no entailment output label")


def classify_with_pytorch(note: str) -> Classification:
    """Zero-shot stage classification with an explicit PyTorch inference loop."""
    tokenizer, model = _load_model()
    entailment_scores = []
    with torch.inference_mode():
        for stage in STAGES:
            tokens = tokenizer(note, HYPOTHESES[stage], return_tensors="pt", truncation=True)
            logits = model(**tokens).logits
            class_probs = torch.softmax(logits, dim=-1)
            entailment_scores.append(class_probs[0, _entailment_index(model)])
    stage_probs = torch.softmax(torch.stack(entailment_scores), dim=0)
    probabilities = {stage: float(prob) for stage, prob in zip(STAGES, stage_probs)}
    label = max(probabilities, key=probabilities.get)
    return Classification(label, probabilities[label], probabilities, "distilbert-nli")


def classify_with_rules(note: str) -> Classification:
    """Offline fallback for explicit stage mentions; not presented as ML."""
    match = re.search(r"\bstage\s*(iv|iii|ii|i|4|3|2|1)\b", note, re.IGNORECASE)
    if not match:
        raise ValueError("Offline fallback requires an explicit stage mention")
    normalized = {"1": "I", "2": "II", "3": "III", "4": "IV"}.get(
        match.group(1), match.group(1).upper()
    )
    probabilities = {stage: float(stage == normalized) for stage in STAGES}
    return Classification(normalized, 1.0, probabilities, "explicit-stage-rule")


def extract_category(note: str, allow_fallback: bool = True) -> Classification:
    try:
        return classify_with_pytorch(note)
    except (OSError, RuntimeError) as exc:
        if not allow_fallback:
            raise
        try:
            return classify_with_rules(note)
        except ValueError:
            raise RuntimeError(
                "The model is unavailable and this note has no explicit stage for the offline fallback"
            ) from exc

