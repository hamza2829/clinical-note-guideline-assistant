from dataclasses import asdict, dataclass

from app.classifier import Classification, extract_category
from app.generator import generate_summary
from app.knowledge import Guideline, lookup_guideline
from app.retrieval import RetrievedPassage, get_retriever


@dataclass(frozen=True)
class PipelineResult:
    classification: Classification
    guideline: Guideline
    passage: RetrievedPassage
    summary: str

    def to_dict(self) -> dict:
        return asdict(self)


def run_pipeline(note: str) -> PipelineResult:
    classification = extract_category(note)
    guideline = lookup_guideline(classification.label)
    passage = get_retriever().retrieve(note)
    summary = generate_summary(note, classification, guideline, passage)
    return PipelineResult(classification, guideline, passage, summary)

