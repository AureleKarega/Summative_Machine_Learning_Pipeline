"""Model creation, testing and retraining.

Data   : data/base_features.csv  (ESC-10, 400 clips, folds 1-5)
         data/uploads_features.csv (clips uploaded through the API/UI, always used for training)
Split  : folds 1-4 + all uploads -> train, fold 5 (never trained on) -> fixed test set,
         so metrics stay comparable before and after every retrain.
"""
import csv, json, os, time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.features import FEATURE_NAMES

ROOT = Path(os.getenv("APP_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))
BASE_CSV = DATA_DIR / "base_features.csv"
UPLOAD_CSV = DATA_DIR / "uploads_features.csv"
UPLOAD_DIR = DATA_DIR / "uploads"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
HISTORY_PATH = MODEL_DIR / "history.json"
STATUS_PATH = MODEL_DIR / "retrain_status.json"
LOCK_PATH = MODEL_DIR / "retrain.lock"
TEST_FOLD = 5
UPLOAD_COLS = ["filename", "label", "fold"] + FEATURE_NAMES


def make_model():
    return make_pipeline(StandardScaler(), SVC(C=10, kernel="rbf", probability=True, random_state=0))


def _atomic_write_json(path, obj):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


# ---------------------------------------------------------------- data
def add_upload(filename, label, features):
    """Append one uploaded clip's features (fold 0 = always train)."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    new = not UPLOAD_CSV.exists()
    with open(UPLOAD_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=UPLOAD_COLS)
        if new:
            w.writeheader()
        w.writerow({"filename": filename, "label": label, "fold": 0, **features})


def load_uploads():
    if UPLOAD_CSV.exists():
        return pd.read_csv(UPLOAD_CSV)
    return pd.DataFrame(columns=UPLOAD_COLS)


def load_dataset():
    base = pd.read_csv(BASE_CSV)
    up = load_uploads()
    if len(up) == 0:
        return base, 0
    return pd.concat([base, up], ignore_index=True), len(up)


# ---------------------------------------------------------------- train
def train(trigger="manual"):
    """Train from scratch on all current data, evaluate on the fixed test fold, save model + metrics."""
    t0 = time.time()
    df, n_uploads = load_dataset()
    train_df = df[df.fold != TEST_FOLD]
    test_df = df[df.fold == TEST_FOLD]

    model = make_model().fit(train_df[FEATURE_NAMES], train_df.label)
    classes = list(model.classes_)
    pred = model.predict(test_df[FEATURE_NAMES])
    proba = model.predict_proba(test_df[FEATURE_NAMES])
    y = test_df.label

    prev = read_json(METRICS_PATH, {})
    metrics = {
        "version": int(prev.get("version", 0)) + 1,
        "model_type": "StandardScaler + SVC(rbf, C=10)",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trigger": trigger,
        "train_seconds": round(time.time() - t0, 2),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_uploads_used": int(n_uploads),
        "classes": classes,
        "accuracy": float(accuracy_score(y, pred)),
        "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, proba, labels=classes)),
        "confusion_matrix": confusion_matrix(y, pred, labels=classes).tolist(),
        "per_class": classification_report(y, pred, output_dict=True, zero_division=0),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = MODEL_PATH.with_suffix(".tmp")
    joblib.dump({"pipeline": model, "features": FEATURE_NAMES, "version": metrics["version"]}, tmp)
    os.replace(tmp, MODEL_PATH)                     # atomic: replicas never see a half-written model
    _atomic_write_json(METRICS_PATH, metrics)
    hist = read_json(HISTORY_PATH, [])
    hist.append({k: metrics[k] for k in ("version", "trained_at", "trigger", "n_train", "n_uploads_used",
                                          "accuracy", "f1_macro", "log_loss", "train_seconds")})
    _atomic_write_json(HISTORY_PATH, hist)
    return metrics


# ---------------------------------------------------------------- shared lock / status (works across containers)
def acquire_lock(stale_after=900):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if LOCK_PATH.exists() and time.time() - LOCK_PATH.stat().st_mtime > stale_after:
            LOCK_PATH.unlink()
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False


def release_lock():
    try:
        LOCK_PATH.unlink()
    except FileNotFoundError:
        pass


def set_status(**kw):
    s = read_json(STATUS_PATH, {})
    s.update(kw)
    _atomic_write_json(STATUS_PATH, s)


def run_retrain_job(trigger):
    """Run a full retrain with locking + status reporting. Returns False if one is already running."""
    if not acquire_lock():
        return False
    try:
        set_status(state="running", trigger=trigger, started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   message="Training...")
        m = train(trigger=trigger)
        set_status(state="done", finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   message=f"Model v{m['version']} trained: accuracy {m['accuracy']:.3f}, F1 {m['f1_macro']:.3f}")
    except Exception as e:                           # noqa: BLE001
        set_status(state="failed", finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), message=str(e))
    finally:
        release_lock()
    return True


def ensure_model():
    if not MODEL_PATH.exists():
        train(trigger="initial")


if __name__ == "__main__":
    m = train(trigger="cli")
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()
           if k in ("version", "accuracy", "precision_macro", "recall_macro", "f1_macro", "log_loss", "n_train", "n_test")})
