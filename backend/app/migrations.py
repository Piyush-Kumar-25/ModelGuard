"""Transactional, versioned additive migrations for existing installations."""
from .database import connect, now

MIGRATIONS = [
    {"datasets": {"role": "TEXT NOT NULL DEFAULT 'reference'", "batch_timestamp": "TEXT", "model_id": "TEXT", "label_availability": "TEXT NOT NULL DEFAULT 'unconfirmed'", "target_column": "TEXT", "record_id_column": "TEXT", "compatibility_json": "TEXT", "mapping_json": "TEXT NOT NULL DEFAULT '{}'", "parent_batch_id": "TEXT"},
     "models": {"status": "TEXT NOT NULL DEFAULT 'legacy'", "parent_id": "TEXT", "checksum": "TEXT", "state_path": "TEXT"},
     "jobs": {"progress": "INTEGER NOT NULL DEFAULT 0", "stage": "TEXT NOT NULL DEFAULT 'queued'", "timeout_seconds": "INTEGER NOT NULL DEFAULT 1800"}}
]

def migrate():
    with connect() as con:
        con.execute("BEGIN IMMEDIATE")
        con.execute("CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
        applied={r[0] for r in con.execute("SELECT version FROM schema_migrations")}
        for version, tables in enumerate(MIGRATIONS, 1):
            if version in applied: continue
            for table, cols in tables.items():
                existing={r[1] for r in con.execute(f"PRAGMA table_info({table})")}
                for name, sql in cols.items():
                    if name not in existing: con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql}")
            con.execute("INSERT INTO schema_migrations VALUES(?,?)",(version,now()))
        con.execute("CREATE INDEX IF NOT EXISTS dataset_project_role ON datasets(project_id,role)")
        con.execute("CREATE INDEX IF NOT EXISTS model_project_status ON models(project_id,status)")
