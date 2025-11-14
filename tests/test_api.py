import json
from app import app


def test_predict_missing_params():
    client = app.test_client()
    resp = client.get('/predict')
    assert resp.status_code == 400


def test_train_and_predict_endpoints_monkeypatch(monkeypatch):
    client = app.test_client()

    # monkeypatch model.model_train to avoid heavy work
    called = {'train': False}

    def fake_train(data_dir, test=False):
        called['train'] = True

    monkeypatch.setattr('model.model_train', fake_train)

    resp = client.post('/train', json={'test': True})
    assert resp.status_code == 200
    assert called['train'] is True

    # monkeypatch model.model_predict to return a simple payload
    def fake_predict(country, year, month, day, all_models=None, test=False):
        return {'y_pred': [123.0], 'y_proba': None}

    monkeypatch.setattr('model.model_predict', fake_predict)

    resp = client.get('/predict?country=all&year=2018&month=01&day=05')
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data['status'] == 'ok'
    assert 'result' in data
