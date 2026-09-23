from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql://postgres:postgres@localhost:5432/clinical_demo"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    classifier_model: str = "typeform/distilbert-base-uncased-mnli"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_postgres: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()

