from pathlib import Path
import shutil
import pytest

import model


class DummyGrid:
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        import numpy as np
        return np.zeros(X.shape[0])


def test_end_to_end_train_predict(monkeypatch):
    # use DummyGrid to keep training quick
    monkeypatch.setattr(model, 'GridSearchCV', DummyGrid)

    repo_root = Path(__file__).resolve().parents[1]
    models_dir = repo_root / 'models'
    if models_dir.exists():
        shutil.rmtree(models_dir)

    data_dir = str(repo_root / 'data' / 'cs-train')
    # train (test mode to limit countries)
    model.model_train(data_dir, test=True)

    # load models and data (recent training saved test-*.joblib files)
    all_data, all_models = model.model_load(prefix='test', data_dir=data_dir, training=False)
    assert 'all' in all_models

    # pick a valid date from data
    dates = all_data['all']['dates']
    if len(dates) == 0:
        pytest.skip('no dates available')
    target = dates[0]
    year, month, day = target.split('-')

    # call model_predict and pass the correct data_dir so model_load uses right path
    res = model.model_predict('all', year, month, day, data_dir=data_dir)
    assert 'y_pred' in res
