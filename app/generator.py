from openai import OpenAI

from app.classifier import Classification
from app.config import get_settings
from app.knowledge import Guideline
from app.retrieval import RetrievedPassage


DISCLAIMER = "Educational toy output using synthetic data; not medical advice or clinically validated."


def generate_summary(
    note: str,
    classification: Classification,
    guideline: Guideline,
    passage: RetrievedPassage,
) -> str:
    settings = get_settings()
    if not settings.openai_api_key:
        return (
            f"Detected stage {classification.label}. {guideline.summary} "
            f"Retrieved context: {passage.passage} {DISCLAIMER}"
        )

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.responses.create(
        model=settings.openai_model,
        instructions=(
            "Write a concise educational summary grounded only in the supplied lookup and passage. "
            "Do not diagnose, prescribe, or add facts. End with the supplied disclaimer."
        ),
        input=(
            f"Synthetic note: {note}\nDetected stage: {classification.label}\n"
            f"Structured lookup: {guideline.summary}\nRetrieved passage: {passage.passage}\n"
            f"Disclaimer: {DISCLAIMER}"
        ),
    )
    return response.output_text

