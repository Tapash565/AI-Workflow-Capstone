from flask import Flask, request, jsonify
import os
import model
import logger

app = Flask(__name__)


@app.route('/train', methods=['POST'])
def train():
    data = request.get_json() or {}
    test_flag = bool(data.get('test', False))
    # Allow overriding data_dir in request for flexibility/tests
    data_dir = data.get('data_dir') or os.path.join('data', 'cs-train')
    # validate data_dir
    if not os.path.isdir(data_dir):
        return jsonify({'status': 'error', 'message': 'data_dir not found'}), 400
    try:
        model.model_train(data_dir, test=test_flag)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    return jsonify({'status': 'ok', 'test': test_flag})


@app.route('/predict', methods=['GET'])
def predict():
    country = request.args.get('country')
    year = request.args.get('year')
    month = request.args.get('month')
    day = request.args.get('day')
    if not all([country, year, month, day]):
        return jsonify({'status': 'error', 'message': 'missing params'}), 400
    # basic validation for date parts
    if not (year.isdigit() and len(year) == 4):
        return jsonify({'status': 'error', 'message': 'invalid year'}), 400
    if not (month.isdigit() and 1 <= int(month) <= 12):
        return jsonify({'status': 'error', 'message': 'invalid month'}), 400
    if not (day.isdigit() and 1 <= int(day) <= 31):
        return jsonify({'status': 'error', 'message': 'invalid day'}), 400
    try:
        # allow caller to pass preloaded models/data via optional params
        res = model.model_predict(country, year, month, day)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    # ensure result uses JSON serializable types
    result = {}
    if isinstance(res, dict):
        for k, v in res.items():
            try:
                # numpy arrays -> lists
                if hasattr(v, 'tolist'):
                    result[k] = v.tolist()
                else:
                    result[k] = v
            except Exception:
                result[k] = str(v)
    else:
        result = str(res)
    return jsonify({'status': 'ok', 'result': result})


@app.route('/logs/<logname>', methods=['GET'])
def logs(logname):
    # return last 20 entries from the named jsonl log
    fname = f"{logname}_log.jsonl"
    path = os.path.join('logs', fname)
    if not os.path.exists(path):
        return jsonify({'status': 'error', 'message': 'log not found'}), 404
    items = []
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(__import__('json').loads(line))
            except Exception:
                continue
    return jsonify({'status': 'ok', 'entries': items[-20:]})


if __name__ == '__main__':
    # run dev server
    app.run(host='0.0.0.0', port=5000, debug=False)
