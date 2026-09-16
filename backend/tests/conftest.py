import os
from pathlib import Path

import pytest


@pytest.fixture()
def client(tmp_path,monkeypatch):
    monkeypatch.setenv("MODELGUARD_DATA_DIR",str(tmp_path/"data"))
    monkeypatch.setenv("MODELGUARD_DB_PATH",str(tmp_path/"data"/"test.db"))
    import app.config as config
    config.settings=config.Settings()
    import app.database as database
    database.settings=config.settings
    database.init_db()
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c: yield c

