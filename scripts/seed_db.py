from pathlib import Path

from psycopg import connect

from app.config import get_settings
from app.retrieval import PgVectorRetriever


def main() -> None:
    settings = get_settings()
    schema = Path("sql/init.sql").read_text(encoding="utf-8")
    with connect(settings.database_url) as connection:
        connection.execute(schema)
    PgVectorRetriever(settings.database_url).seed()
    print("Seeded guideline passages.")


if __name__ == "__main__":
    main()

