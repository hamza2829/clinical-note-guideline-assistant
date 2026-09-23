from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from pgvector.psycopg import register_vector
from psycopg import connect
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.knowledge import PASSAGES


@dataclass(frozen=True)
class RetrievedPassage:
    stage: str
    passage: str
    source_url: str
    similarity: float


@lru_cache
def _embedder() -> SentenceTransformer:
    return SentenceTransformer(get_settings().embedding_model)


def _embed(texts: list[str]) -> np.ndarray:
    return _embedder().encode(texts, normalize_embeddings=True)


class InMemoryRetriever:
    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None

    def retrieve(self, query: str) -> RetrievedPassage:
        if self._vectors is None:
            self._vectors = _embed([item["passage"] for item in PASSAGES])
        query_vector = _embed([query])[0]
        similarities = self._vectors @ query_vector
        index = int(np.argmax(similarities))
        item = PASSAGES[index]
        return RetrievedPassage(**item, similarity=float(similarities[index]))


class PgVectorRetriever:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def seed(self) -> None:
        vectors = _embed([item["passage"] for item in PASSAGES])
        with connect(self.database_url) as connection:
            register_vector(connection)
            with connection.cursor() as cursor:
                for item, vector in zip(PASSAGES, vectors):
                    cursor.execute(
                        """INSERT INTO guideline_passages(stage, passage, source_url, embedding)
                           VALUES (%s, %s, %s, %s)
                           ON CONFLICT (passage) DO UPDATE SET embedding = EXCLUDED.embedding""",
                        (item["stage"], item["passage"], item["source_url"], vector),
                    )

    def retrieve(self, query: str) -> RetrievedPassage:
        query_vector = _embed([query])[0]
        with connect(self.database_url) as connection:
            register_vector(connection)
            row = connection.execute(
                """SELECT stage, passage, source_url, 1 - (embedding <=> %s) AS similarity
                   FROM guideline_passages ORDER BY embedding <=> %s LIMIT 1""",
                (query_vector, query_vector),
            ).fetchone()
        if row is None:
            raise RuntimeError("No passages found; run `python -m scripts.seed_db`")
        return RetrievedPassage(*row)


@lru_cache
def get_retriever():
    settings = get_settings()
    if settings.use_postgres:
        return PgVectorRetriever(settings.database_url)
    return InMemoryRetriever()

