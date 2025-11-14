import os
import joblib
from pathlib import Path
import shutil
import numpy as np

import model


class DummyGrid:
    def __init__(self, *args, **kwargs):
        self.fitted = False

    def fit(self, X, y):
        # pretend to fit
        self.fitted = True
        return self

    def predict(self, X):
        # return zeros matching X.shape[0]
        return np.zeros(X.shape[0])


def test__model_train_with_dummy_grid(monkeypatch, tmp_path):
    # monkeypatch GridSearchCV in model to DummyGrid
    monkeypatch.setattr(model, 'GridSearchCV', DummyGrid)

    # ensure models dir
    repo_root = Path(__file__).resolve().parents[1]
    models_dir = repo_root / 'models'
    if models_dir.exists():
        shutil.rmtree(models_dir)
    models_dir.mkdir()

    # load a small ts dataset from cslib via model.model_train pathways
    data_dir = str(Path(__file__).resolve().parents[1] / 'data' / 'cs-train')

    # run training in test mode (will save test-<tag> models)
    model.model_train(data_dir, test=True)

    # ensure at least one model saved
    files = list(models_dir.glob('test-*.joblib'))
    assert len(files) > 0
