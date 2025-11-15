#!/usr/bin/env python
"""Generate a model vs actual plot for a country and save PNG + metrics.

This script loads available models (test or production), loads the
corresponding data via `model.model_load`, predicts on the last N dates
and writes `reports/model_vs_actual_<country>.png` and a JSON metrics file.
"""
import os
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

# ensure repo root is on sys.path so imports work when running from scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import model


def main(country='all', n_points=30, data_dir=None):
    data_dir = data_dir or os.path.join(str(REPO_ROOT), 'data', 'cs-train')
    # try loading test models first, fall back to production
    prefixes = ['test', 'sl']
    try:
        all_data, all_models = model.model_load(prefix=prefixes, data_dir=data_dir, training=False)
        if country not in all_models:
            raise Exception('no_model')
        data = all_data[country]
        mdl = all_models[country]
    except Exception:
        # fallback: use DummyModel and load ts data directly
        print('Could not load serialized model(s); falling back to DummyModel for plotting')
        from scripts.dummy_model import DummyModel
        from cslib import fetch_ts, engineer_features

        ts = fetch_ts(data_dir)
        if country not in ts:
            raise SystemExit(f"No data for country={country}")
        df = ts[country]
        X, y, dates = engineer_features(df, training=False)
        # align types similar to model_load output
        data = {'X': X, 'y': y, 'dates': dates}
        mdl = DummyModel()

    # get last n_points rows
    total = data['X'].shape[0]
    n = min(n_points, total)
    idxs = np.arange(total)[-n:]
    queries = data['X'].iloc[idxs]
    preds = mdl.predict(queries)
    actuals = data['y'][idxs]
    dates = data['dates'][idxs]
    dates_str = [str(d) for d in dates]

    out_dir = REPO_ROOT / 'reports'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f'model_vs_actual_{country}.png'
    out_json = out_dir / f'model_vs_actual_{country}.json'

    plt.figure(figsize=(10,4))
    plt.plot(dates_str, actuals, label='actual', marker='o')
    plt.plot(dates_str, preds, label='predicted', marker='x')
    plt.xticks(rotation=45)
    plt.xlabel('date')
    plt.ylabel('30-day revenue')
    plt.title(f'Model vs Actual ({country})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    # compute RMSE
    rmse = float(np.sqrt(np.mean((np.array(preds) - np.array(actuals)) ** 2)))
    metrics = {
        'country': country,
        'pairs': int(n),
        'rmse': rmse
    }
    with open(out_json, 'w', encoding='utf-8') as fh:
        json.dump(metrics, fh, indent=2)

    print('Wrote', out_png)
    print('Wrote', out_json)


if __name__ == '__main__':
    main()
