# Clinical Note Extraction & Guideline Lookup Assistant

An entry-level NLP/PyTorch/RAG learning project. It runs one transparent pipeline:

```text
synthetic note
  -> DistilBERT zero-shot stage classification (plain PyTorch inference)
  -> stage-to-guideline dictionary lookup
  -> semantic passage retrieval (pgvector or local cosine search)
  -> one optional LLM call for concise phrasing
```

> **Important:** This project accepts synthetic examples only. It is not clinically validated, does not diagnose or recommend care, and must not be used with protected health information or for medical decisions.

## What it demonstrates

- A small pretrained `typeform/distilbert-base-uncased-mnli` classifier loaded with Hugging Face, with the tokenization, forward pass, and `torch.softmax` inference loop written directly in `app/classifier.py`.
- A deliberately small Python dictionary mapping breast-cancer stages I–IV to educational treatment summaries.
- Retrieval over four guideline snippets using MiniLM embeddings and either PostgreSQL/pgvector or an in-memory cosine-search fallback.
- A FastAPI endpoint and a linear, testable pipeline—not a multi-agent system.
- An evaluation script over 10 clearly labeled synthetic notes.

The model is used zero-shot; it is **not fine-tuned**, and this repository makes no custom-model or clinical-performance claim.

## Quick start

Requires Python 3.11 (recommended).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs`, or call:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/analyze `
  -ContentType 'application/json' `
  -Body '{"note":"Synthetic record: stage IV metastatic breast cancer with distant spread."}'
```

On first use, Hugging Face downloads the classifier and embedding models. If the classifier is unavailable, the demo transparently falls back to a regex **only for explicit stage mentions** and reports `explicit-stage-rule` as its method. The fallback is not represented as machine learning.

## PostgreSQL + pgvector

Local cosine retrieval is the zero-setup default. To exercise the database-backed path:

```powershell
docker compose up -d
$env:USE_POSTGRES='true'
python -m scripts.seed_db
uvicorn app.main:app --reload
```

The SQL schema uses a 384-dimensional vector and an HNSW cosine index. Set `DATABASE_URL` if your connection differs from `.env.example`.

## Optional final LLM call

Without `OPENAI_API_KEY`, the pipeline returns a deterministic grounded template. If a key is provided, it makes exactly one Responses API call after classification, lookup, and retrieval. The prompt restricts the model to supplied context and requires the disclaimer. This call is presentation-only; it does not decide the category or select the guideline.

## Evaluation and tests

```powershell
pytest -q
python -m scripts.evaluate
```

The evaluation writes `outputs/evaluation_report.json` and scores both stage classification and retrieved-stage match. All 10 inputs contain explicit stage wording by design. Consequently, this measures wiring and repeatability, not real clinical NLP performance. A serious evaluation would need expert-reviewed, de-identified data, subgroup/error analysis, and governance review.

## Data flow and limitations

The API returns the chosen label, confidence distribution, method, structured lookup, retrieved source URL and similarity, and final summary. The components remain separate so failures can be inspected rather than hidden behind generated prose.

Known limitations include the tiny synthetic test set, simplified stage labels, a deliberately compressed guideline table, no negation/temporality handling, no calibration study, and no protection suitable for clinical deployment. Treatment varies with subtype, biomarkers, prior response, health, preferences, and other factors that this toy mapping does not model.

## Sources

The educational summaries and passages are short paraphrases of these public National Cancer Institute pages, accessed September 23, 2026:

- [Treatment of Breast Cancer by Stage](https://www.cancer.gov/types/breast/treatment/by-stage)
- [Breast Cancer Stages](https://www.cancer.gov/types/breast/stages)

The repository stores paraphrases and source URLs, not scraped clinical records.

