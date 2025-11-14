import os
import json
from datetime import datetime, timezone


def _ensure_logs_dir(test=False):
    d = "logs-test" if test else "logs"
    os.makedirs(d, exist_ok=True)
    return d


def _utc_ts():
    return datetime.now(timezone.utc).isoformat()


def update_train_log(tag, date_range, metrics, runtime, version, note, test=False):
    """Write a training log entry as a JSON line.

    This switches logs to newline-delimited JSON for easier parsing and
    includes an ISO8601 UTC timestamp.
    """
    d = _ensure_logs_dir(test)
    file_path = os.path.join(d, "train_log.jsonl")
    entry = {
        "timestamp": _utc_ts(),
        "tag": tag,
        "start_date": date_range[0],
        "end_date": date_range[1],
        "metrics": metrics,
        "runtime": runtime,
        "version": version,
        "note": note,
    }
    with open(file_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str))
        fh.write("\n")


def update_predict_log(country, y_pred, y_proba, target_date, runtime, version, test=False):
    """Write a predict log entry as a JSON line.
    """
    d = _ensure_logs_dir(test)
    file_path = os.path.join(d, "predict_log.jsonl")
    entry = {
        "timestamp": _utc_ts(),
        "country": country,
        "target_date": target_date,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "runtime": runtime,
        "version": version,
    }
    with open(file_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str))
        fh.write("\n")


if __name__ == "__main__":
    # quick smoke test
    update_train_log("test", ("2020-01-01", "2020-02-01"), {"rmse": 1.23}, "00:00:01", 0.1, "note", test=True)
    update_predict_log("all", [10.0], None, "2020-01-02", "00:00:00", 0.1, test=True)
