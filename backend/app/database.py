from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import settings

_lock = threading.RLock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect() -> sqlite3.Connection:
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(settings.db_path, timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    return con


def init_db() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with connect() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS projects(
          id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT,
          config_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS datasets(
          id TEXT PRIMARY KEY, project_id TEXT NOT NULL, original_name TEXT NOT NULL,
          path TEXT NOT NULL, sha256 TEXT NOT NULL, rows INTEGER NOT NULL, columns INTEGER NOT NULL,
          schema_json TEXT NOT NULL, created_at TEXT NOT NULL,
          FOREIGN KEY(project_id) REFERENCES projects(id));
        CREATE TABLE IF NOT EXISTS models(
          id TEXT PRIMARY KEY, project_id TEXT NOT NULL, dataset_id TEXT NOT NULL,
          version INTEGER NOT NULL, path TEXT NOT NULL, task TEXT NOT NULL, selected_name TEXT NOT NULL,
          metrics_json TEXT NOT NULL, details_json TEXT NOT NULL, created_at TEXT NOT NULL,
          FOREIGN KEY(project_id) REFERENCES projects(id));
        CREATE TABLE IF NOT EXISTS analyses(
          id TEXT PRIMARY KEY, project_id TEXT NOT NULL, kind TEXT NOT NULL,
          result_json TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS experiments(
          id TEXT PRIMARY KEY, project_id TEXT NOT NULL, experiment_type TEXT NOT NULL,
          params_json TEXT NOT NULL, result_json TEXT NOT NULL, random_seed INTEGER NOT NULL,
          created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs(
          id TEXT PRIMARY KEY, project_id TEXT, kind TEXT NOT NULL, status TEXT NOT NULL,
          result_json TEXT, error_json TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        """)


def insert(table: str, values: dict[str, Any]) -> dict[str, Any]:
    cols = list(values)
    with _lock, connect() as con:
        con.execute(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})", [values[c] for c in cols])
    return values


def one(sql: str, args: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    with connect() as con:
        row = con.execute(sql, args).fetchone()
    return dict(row) if row else None


def all_rows(sql: str, args: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with connect() as con:
        return [dict(r) for r in con.execute(sql, args).fetchall()]


def update(sql: str, args: tuple[Any, ...]) -> None:
    with _lock, connect() as con:
        con.execute(sql, args)


def uid() -> str:
    return str(uuid.uuid4())


def dump(value: Any) -> str:
    return json.dumps(value, allow_nan=False, default=str)


def load(value: str | None, default: Any = None) -> Any:
    return json.loads(value) if value else default


def project_dir(project_id: str) -> Path:
    path = (settings.data_dir / "projects" / project_id).resolve()
    root = (settings.data_dir / "projects").resolve()
    if root not in path.parents:
        raise ValueError("Invalid project path")
    path.mkdir(parents=True, exist_ok=True)
    return path

