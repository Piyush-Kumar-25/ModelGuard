from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import settings
from ..database import dump, insert, load, now, one, project_dir, uid, update
from ..errors import ModelGuardError, require


def get_project(project_id: str) -> dict[str, Any]:
    return require(one("SELECT * FROM projects WHERE id=?", (project_id,)), "Project not found")


def get_dataset(project_id: str, dataset_id: str | None = None) -> dict[str, Any]:
    if dataset_id:
        row = one("SELECT * FROM datasets WHERE id=? AND project_id=?", (dataset_id, project_id))
    else:
        row = one("SELECT * FROM datasets WHERE project_id=? ORDER BY created_at DESC LIMIT 1", (project_id,))
    return require(row, "Dataset not found")


def load_frame(project_id: str, dataset_id: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta = get_dataset(project_id, dataset_id)
    path = Path(meta["path"])
    if not path.is_file():
        raise ModelGuardError("Stored dataset is unavailable", "artifact_missing", 500)
    return pd.read_csv(path), meta


def infer_task(y: pd.Series) -> str:
    unique = y.nunique(dropna=True)
    if (not pd.api.types.is_numeric_dtype(y)) or unique <= max(20, int(len(y) * 0.05)):
        return "classification"
    return "regression"


def create_project(name: str, description: str | None) -> dict[str, Any]:
    pid = uid()
    row = {"id": pid, "name": name.strip(), "description": description, "config_json": "{}", "created_at": now()}
    insert("projects", row)
    project_dir(pid)
    return {**row, "config": {}}


def _safe_filename(name: str) -> str:
    base = Path(name).name
    if base != name or not re.fullmatch(r"[A-Za-z0-9._ -]+", base or "") or Path(base).suffix.lower() != ".csv":
        raise ModelGuardError("Only safe .csv filenames are accepted", "invalid_filename")
    return base


def upload_dataset(project_id: str, filename: str, content: bytes) -> dict[str, Any]:
    get_project(project_id)
    safe = _safe_filename(filename)
    if not content:
        raise ModelGuardError("The uploaded file is empty", "empty_dataset")
    if len(content) > settings.max_upload_bytes:
        raise ModelGuardError("File exceeds configured upload limit", "file_too_large", 413)
    try:
        df = pd.read_csv(io.BytesIO(content), nrows=settings.max_csv_rows + 1)
    except Exception as exc:
        raise ModelGuardError("CSV is malformed or cannot be decoded", "malformed_csv", details=str(exc)) from None
    if len(df) > settings.max_csv_rows:
        raise ModelGuardError("Dataset exceeds configured row limit", "too_many_rows", 413)
    if df.empty or len(df.columns) == 0:
        raise ModelGuardError("Dataset must contain rows and columns", "empty_dataset")
    if len(set(map(str, df.columns))) != len(df.columns):
        raise ModelGuardError("Duplicate column names are not supported", "duplicate_columns")
    did = uid()
    folder = project_dir(project_id) / "datasets" / did
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / "original.csv"
    path.write_bytes(content)  # immutable original; subsequent operations only read it
    schema = [{"name": str(c), "dtype": str(df[c].dtype), "missing": int(df[c].isna().sum()), "unique": int(df[c].nunique(dropna=True))} for c in df.columns]
    row = {"id": did, "project_id": project_id, "original_name": safe, "path": str(path), "sha256": hashlib.sha256(content).hexdigest(), "rows": len(df), "columns": len(df.columns), "schema_json": dump(schema), "created_at": now()}
    insert("datasets", row)
    return {**row, "schema": schema, "schema_json": None}


def summarize(project_id: str, dataset_id: str) -> dict[str, Any]:
    df, meta = load_frame(project_id, dataset_id)
    numeric = df.select_dtypes(include=np.number)
    stats = {}
    for c in df.columns:
        s = df[c]
        item = {"dtype": str(s.dtype), "count": int(s.count()), "missing": int(s.isna().sum()), "unique": int(s.nunique(dropna=True))}
        if c in numeric:
            item.update({k: (None if pd.isna(v) else float(v)) for k, v in s.describe()[["mean", "std", "min", "max"]].items()})
        else:
            top = s.value_counts(dropna=True).head(5)
            item["top_values"] = {str(k): int(v) for k, v in top.items()}
        stats[str(c)] = item
    return {"dataset_id": dataset_id, "rows": len(df), "columns": list(map(str, df.columns)), "inferred_schema": load(meta["schema_json"]), "statistics": stats}


def configure(project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    project = get_project(project_id)
    df, _ = load_frame(project_id, payload["dataset_id"])
    target = payload["target_column"]
    if target not in df.columns:
        raise ModelGuardError("Target column does not exist", "invalid_target")
    if df[target].isna().all() or df[target].nunique(dropna=True) < 2:
        raise ModelGuardError("Target must contain at least two non-null values", "invalid_target")
    if payload.get("time_column") and payload["time_column"] not in df.columns:
        raise ModelGuardError("Time column does not exist", "invalid_time_column")
    payload["task"] = infer_task(df[target]) if payload.get("task") == "auto" else payload["task"]
    allowed = {"classification": {"macro_f1", "weighted_f1", "f1", "roc_auc", "pr_auc"}, "regression": {"rmse", "mae", "r2"}}
    payload["primary_metric"] = payload.get("primary_metric") or ("macro_f1" if payload["task"] == "classification" else "rmse")
    if payload["primary_metric"] not in allowed[payload["task"]]:
        raise ModelGuardError("Primary metric is incompatible with task", "invalid_metric")
    update("UPDATE projects SET config_json=? WHERE id=?", (dump(payload), project_id))
    return payload


def configured(project_id: str) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    p = get_project(project_id)
    config = load(p["config_json"], {})
    if not config.get("dataset_id"):
        raise ModelGuardError("Configure a dataset and target first", "project_not_configured")
    df, meta = load_frame(project_id, config["dataset_id"])
    return config, df, meta

