from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("MODELGUARD_DATA_DIR", "data")).resolve()
    db_path: Path = Path(os.getenv("MODELGUARD_DB_PATH", "data/modelguard.db")).resolve()
    max_upload_bytes: int = int(os.getenv("MODELGUARD_MAX_UPLOAD_MB", "50")) * 1024 * 1024
    max_csv_rows: int = int(os.getenv("MODELGUARD_MAX_CSV_ROWS", "1000000"))
    random_seed: int = int(os.getenv("MODELGUARD_RANDOM_SEED", "42"))
    test_size: float = float(os.getenv("MODELGUARD_TEST_SIZE", "0.20"))
    min_region_samples: int = int(os.getenv("MODELGUARD_MIN_REGION_SAMPLES", "50"))
    min_region_fraction: float = float(os.getenv("MODELGUARD_MIN_REGION_FRACTION", "0.05"))


settings = Settings()

