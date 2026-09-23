from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.pipeline import run_pipeline


app = FastAPI(
    title="Clinical Note Extraction & Guideline Lookup (Toy Demo)",
    description="Synthetic educational project. Not medical advice or clinically validated.",
    version="0.1.0",
)


class AnalyzeRequest(BaseModel):
    note: str = Field(min_length=10, max_length=5000)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    try:
        return run_pipeline(request.note).to_dict()
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

