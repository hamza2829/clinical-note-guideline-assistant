CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS guideline_passages (
    id BIGSERIAL PRIMARY KEY,
    stage TEXT NOT NULL,
    passage TEXT NOT NULL UNIQUE,
    source_url TEXT NOT NULL,
    embedding vector(384) NOT NULL
);

CREATE INDEX IF NOT EXISTS guideline_passages_embedding_idx
ON guideline_passages USING hnsw (embedding vector_cosine_ops);

