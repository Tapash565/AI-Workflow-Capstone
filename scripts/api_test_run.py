import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import app
import json

client = app.test_client()

# call predict
resp = client.get('/predict?country=all&year=2017&month=11&day=29')
print('PREDICT STATUS:', resp.status_code)
try:
    print(json.dumps(resp.get_json(), indent=2))
except Exception:
    print(resp.data)

# call logs (may not exist)
resp2 = client.get('/logs/predict')
print('\nLOGS STATUS:', resp2.status_code)
try:
    print(json.dumps(resp2.get_json(), indent=2)[:1000])
except Exception:
    print(resp2.data)
