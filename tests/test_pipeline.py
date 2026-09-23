from app.classifier import Classification
from app.knowledge import GUIDELINES
from app.pipeline import run_pipeline
from app.retrieval import RetrievedPassage


class FakeRetriever:
    def retrieve(self, query: str) -> RetrievedPassage:
        return RetrievedPassage("II", "Retrieved stage II context.", "https://example.test", 0.9)


def test_linear_pipeline(monkeypatch) -> None:
    classification = Classification("II", 0.8, {"I": 0.1, "II": 0.8, "III": 0.05, "IV": 0.05}, "test")
    monkeypatch.setattr("app.pipeline.extract_category", lambda note: classification)
    monkeypatch.setattr("app.pipeline.get_retriever", lambda: FakeRetriever())
    monkeypatch.setattr("app.pipeline.generate_summary", lambda *args: "safe summary")

    result = run_pipeline("A sufficiently long synthetic note.")

    assert result.classification.label == "II"
    assert result.guideline == GUIDELINES["II"]
    assert result.passage.stage == "II"
    assert result.summary == "safe summary"

