import os
from pathlib import Path
import shutil
import json

import logger


def test_update_train_log_creates_file(tmp_path):
    logs_dir = Path("logs-test")
    if logs_dir.exists():
        shutil.rmtree(logs_dir)

    logger.update_train_log("unit-test", ("2019-01-01", "2019-02-01"), {"rmse": 0.1}, "00:00:01", 0.1, "note", test=True)

    fpath = logs_dir / "train_log.jsonl"
    assert fpath.exists()
    # verify it's valid JSON lines
    lines = [l for l in fpath.read_text(encoding='utf-8').splitlines() if l.strip()]
    assert len(lines) >= 1
    entry = json.loads(lines[-1])
    assert entry.get("tag") == "unit-test"


def test_update_predict_log_creates_file(tmp_path):
    logs_dir = Path("logs-test")
    if logs_dir.exists():
        shutil.rmtree(logs_dir)

    logger.update_predict_log("all", [1.0], None, "2019-01-02", "00:00:00", 0.1, test=True)
    fpath = logs_dir / "predict_log.jsonl"
    assert fpath.exists()
    lines = [l for l in fpath.read_text(encoding='utf-8').splitlines() if l.strip()]
    assert len(lines) >= 1
    entry = json.loads(lines[-1])
    assert entry.get("country") == "all"
