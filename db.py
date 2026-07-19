from __future__ import annotations

import os

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

DB_PATH = os.getenv("DB_PATH", os.path.join(os.path.dirname(__file__), "storage", "app.db"))
DB_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def _add_missing_columns() -> None:
    """create_all() only creates missing tables, not missing columns on
    tables that already exist. This project has no migration tool, so for
    this sqlite database we patch newly added nullable columns in place."""
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.tables.values():
            if table.name not in inspector.get_table_names():
                continue
            existing_columns = {col["name"] for col in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing_columns:
                    continue
                column_type = column.type.compile(dialect=engine.dialect)
                conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {column_type}'))


def init_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    Base.metadata.create_all(bind=engine)
    _add_missing_columns()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
