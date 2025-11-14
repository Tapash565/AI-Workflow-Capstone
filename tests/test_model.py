import os
import shutil
from pathlib import Path
import pytest

import model


def test_model_load_raises_when_no_models(tmp_path):
    # ensure models dir is absent
    repo_root = Path(__file__).resolve().parents[1]
    models_dir = repo_root / "models"
    if models_dir.exists():
        # move aside to avoid deleting user's models
        tmp_store = repo_root / "models_backup_for_tests"
        if tmp_store.exists():
            shutil.rmtree(tmp_store)
        models_dir.rename(tmp_store)
        moved = True
    else:
        moved = False

    try:
        with pytest.raises(Exception):
            model.model_load()
    finally:
        # restore moved models if any
        if moved:
            tmp_store.rename(models_dir)
