#!/usr/bin/env python
"""Monitoring script: compare logged predictions to actuals and plot results.

Usage:
    python scripts/monitor.py --country all --data-dir data/cs-train --logs logs --out reports/monitor-all.png

This script:
 - reads `predict_log.jsonl` from the logs directory
 - for each entry, finds the matching actual target using `cslib.engineer_features`
 - computes RMSE and saves a time-series plot of predicted vs actual values
"""
import argparse
import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from cslib import fetch_ts, convert_to_ts, engineer_features


def load_predict_log(logs_dir):
    path = Path(logs_dir) / 'predict_log.jsonl'
    if not path.exists():
        raise FileNotFoundError(f"predict log not found at {path}")
    entries = []
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
    return entries


def get_actual_for_date(data_dir, country, target_date):
    # fetch ts data for the country
    ts = fetch_ts(data_dir)
    if country not in ts:
        raise Exception(f"country '{country}' not available in ts data")
    df = ts[country]
    X, y, dates = engineer_features(df, training=False)
    dates = np.array([str(d) for d in dates])
    if target_date not in dates:
        return None
    idx = np.where(dates == target_date)[0][0]
    return float(y[idx])


def plot_results(dates, y_true, y_pred, out_path, country):
    plt.figure(figsize=(10,4))
    plt.plot(dates, y_true, label='actual', marker='o')
    plt.plot(dates, y_pred, label='predicted', marker='x')
    plt.xticks(rotation=45)
    plt.xlabel('date')
    plt.ylabel('30-day revenue')
    plt.title(f'Predicted vs Actual (country={country})')
    plt.legend()
    plt.tight_layout()
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--country', default='all')
    p.add_argument('--data-dir', default=os.path.join('data','cs-train'))
    p.add_argument('--logs', default='logs')
    p.add_argument('--out', default=os.path.join('reports', 'monitor-{}.png'))
    args = p.parse_args()

    entries = load_predict_log(args.logs)
    # filter entries for country
    filtered = [e for e in entries if e.get('country') == args.country]
    if not filtered:
        print('No predictions found for country:', args.country)
        return

    dates = []
    preds = []
    actuals = []
    missing = 0
    for e in filtered:
        target_date = e.get('target_date')
        y_pred = e.get('y_pred')
        # y_pred might be a list/str; normalize
        if isinstance(y_pred, list):
            pred_val = float(y_pred[0])
        else:
            # try parse
            try:
                pred_val = float(y_pred)
            except Exception:
                try:
                    arr = json.loads(y_pred)
                    pred_val = float(arr[0])
                except Exception:
                    pred_val = None

        if pred_val is None:
            continue

        actual = get_actual_for_date(args.data_dir, args.country, target_date)
        if actual is None:
            missing += 1
            continue

        dates.append(target_date)
        preds.append(pred_val)
        actuals.append(actual)

    if not dates:
        print('No matching actuals found for predictions (missing {})'.format(missing))
        return

    dates = np.array(dates)
    preds = np.array(preds)
    actuals = np.array(actuals)

    rmse = np.sqrt(np.mean((preds - actuals) ** 2))
    print(f'Found {len(dates)} pairs. RMSE={rmse:.3f}')

    out_path = args.out.format(args.country)
    plot_results(dates, actuals, preds, out_path, args.country)
    print('Saved plot to', out_path)


if __name__ == '__main__':
    main()
