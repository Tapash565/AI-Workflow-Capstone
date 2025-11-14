import os
from pathlib import Path
import shutil

import cslib


def test_fetch_data_reads_json():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "cs-train"
    assert data_dir.exists()

    df = cslib.fetch_data(str(data_dir))
    # expect columns to exist
    for col in ["country", "invoice", "price", "year", "month", "day"]:
        assert col in df.columns


def test_fetch_ts_creates_ts_files(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "cs-train"
    ts_dir = data_dir / "ts-data"
    # remove if exists to test clean creation
    if ts_dir.exists():
        shutil.rmtree(ts_dir)

    dfs = cslib.fetch_ts(str(data_dir), clean=False)
    # must contain 'all' dataset
    assert "all" in dfs
    assert (ts_dir / "ts-all.csv").exists()
