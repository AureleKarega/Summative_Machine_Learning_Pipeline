"""FastAPI service: prediction, bulk upload, retraining trigger, monitoring + dashboard UI."""
import os, pickle, re, threading, time, uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from src import train as T
from src.features import FEATURE_NAMES, extract_features, load_audio

AUTO_RETRAIN_THRESHOLD = int(os.getenv("AUTO_RETRAIN_THRESHOLD", "20"))   # pending uploads that auto-trigger retraining
MAX_FILE_MB = 10
STARTED = time.time()
STARTED_AT = datetime.now(timezone.utc).isoformat(timespec="seconds")
STATS = {"predictions": 0, "conf_sum": 0.0, "errors": 0}
_cache = {"mtime": None, "bundle": None}
_lock = threading.Lock()

app = FastAPI(title="ESC-10 Sound Classifier API", version="1.0")


@app.on_event("startup")
def _startup():
    T.ensure_model()          # first boot: train from base_features.csv if no model exists


def get_bundle():
    """Return the loaded model, reloading automatically when another container retrains it."""
    mtime = T.MODEL_PATH.stat().st_mtime
    with _lock:
        if _cache["mtime"] != mtime:
            with open(T.MODEL_PATH, "rb") as f:
                _cache["bundle"] = pickle.load(f)
            _cache["mtime"] = mtime
        return _cache["bundle"]


def pending_uploads():
    total = len(T.load_uploads())
    used = T.read_json(T.METRICS_PATH, {}).get("n_uploads_used", 0)
    return total, max(total - used, 0)


# ------------------------------------------------------------ monitoring
@app.get("/api/health")
def health():
    m = T.read_json(T.METRICS_PATH, {})
    return {"status": "ok", "uptime_seconds": round(time.time() - STARTED, 1), "started_at": STARTED_AT,
            "model_version": m.get("version"), "model_trained_at": m.get("trained_at"),
            "predictions_served": STATS["predictions"],
            "avg_confidence": round(STATS["conf_sum"] / STATS["predictions"], 3) if STATS["predictions"] else None,
            "container": os.getenv("HOSTNAME", "local")}


@app.get("/api/metrics")
def metrics():
    return {"current": T.read_json(T.METRICS_PATH, {}), "history": T.read_json(T.HISTORY_PATH, [])}


# ------------------------------------------------------------ prediction
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {MAX_FILE_MB} MB")
    try:
        feats = await run_in_threadpool(lambda: extract_features(load_audio(data)))
    except Exception as e:                                         # noqa: BLE001
        STATS["errors"] += 1
        raise HTTPException(400, f"Could not read audio: {e}")
    b = get_bundle()
    proba = b["pipeline"].predict_proba(pd.DataFrame([feats])[FEATURE_NAMES])[0]
    order = proba.argsort()[::-1]
    classes = b["pipeline"].classes_
    STATS["predictions"] += 1
    STATS["conf_sum"] += float(proba[order[0]])
    return {"prediction": str(classes[order[0]]), "confidence": round(float(proba[order[0]]), 4),
            "top3": [{"label": str(classes[i]), "probability": round(float(proba[i]), 4)} for i in order[:3]],
            "model_version": b["version"]}


# ------------------------------------------------------------ upload + retrain
def _slug(label):
    s = re.sub(r"[^a-z0-9_]+", "_", label.strip().lower()).strip("_")
    if not s:
        raise HTTPException(400, "Label is required (letters, numbers, underscores)")
    return s


@app.post("/api/upload")
async def upload(background: BackgroundTasks, label: str = Form(...), files: list[UploadFile] = File(...)):
    """Bulk upload of labelled .wav files. Features are extracted immediately; retraining is triggered
    automatically once AUTO_RETRAIN_THRESHOLD new clips are waiting."""
    label = _slug(label)
    accepted, rejected = 0, []
    folder = T.UPLOAD_DIR / label
    folder.mkdir(parents=True, exist_ok=True)
    for f in files:
        try:
            if not f.filename.lower().endswith(".wav"):
                raise ValueError("only .wav files are accepted")
            data = await f.read()
            if len(data) > MAX_FILE_MB * 1024 * 1024:
                raise ValueError(f"larger than {MAX_FILE_MB} MB")
            feats = await run_in_threadpool(lambda: extract_features(load_audio(data)))
            name = f"{uuid.uuid4().hex[:8]}_{Path(f.filename).name}"
            (folder / name).write_bytes(data)
            T.add_upload(name, label, feats)
            accepted += 1
        except Exception as e:                                     # noqa: BLE001
            rejected.append({"file": f.filename, "reason": str(e)})
    total, pending = pending_uploads()
    auto = False
    if pending >= AUTO_RETRAIN_THRESHOLD and T.read_json(T.STATUS_PATH, {}).get("state") != "running":
        background.add_task(T.run_retrain_job, "auto: upload threshold")
        auto = True
    return {"accepted": accepted, "rejected": rejected, "label": label, "pending_uploads": pending,
            "auto_retrain_threshold": AUTO_RETRAIN_THRESHOLD, "auto_retrain_started": auto}


@app.post("/api/retrain")
def retrain(background: BackgroundTasks):
    """Manual retraining trigger (UI button). Also callable by a scheduler / webhook."""
    if T.read_json(T.STATUS_PATH, {}).get("state") == "running" and T.LOCK_PATH.exists():
        return JSONResponse({"started": False, "message": "A retraining job is already running"}, status_code=409)
    background.add_task(T.run_retrain_job, "manual")
    return {"started": True, "message": "Retraining started"}


@app.get("/api/retrain/status")
def retrain_status():
    total, pending = pending_uploads()
    return {**T.read_json(T.STATUS_PATH, {"state": "idle"}), "total_uploads": total, "pending_uploads": pending,
            "auto_retrain_threshold": AUTO_RETRAIN_THRESHOLD}


# ------------------------------------------------------------ data for the dashboard
@app.get("/api/visuals")
def visuals():
    df, _ = T.load_dataset()
    counts = df.label.value_counts().sort_index()
    grp = df.groupby("label")[FEATURE_NAMES]
    return {"class_counts": counts.to_dict(),
            "feature_means": grp.mean().round(5).to_dict(orient="index"),
            "feature_stds": grp.std().fillna(0).round(5).to_dict(orient="index"),
            "features": FEATURE_NAMES}


@app.get("/api/scatter")
def scatter(x: str = "spectral_centroid_mean", y: str = "rms_mean"):
    if x not in FEATURE_NAMES or y not in FEATURE_NAMES:
        raise HTTPException(400, "Unknown feature")
    df, _ = T.load_dataset()
    return {"x": x, "y": y, "points": {k: g[[x, y]].round(5).values.tolist() for k, g in df.groupby("label")}}


app.mount("/", StaticFiles(directory=Path(__file__).resolve().parents[1] / "ui", html=True), name="ui")
