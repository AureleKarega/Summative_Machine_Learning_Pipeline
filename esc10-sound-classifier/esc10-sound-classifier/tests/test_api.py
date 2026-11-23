"""Smoke tests. Run: pytest -q   (or python tests/test_api.py)"""
import os, sys, tempfile, shutil, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
tmp = tempfile.mkdtemp()
shutil.copytree(ROOT / "data", Path(tmp) / "data"); os.makedirs(Path(tmp) / "models")
os.environ.update(DATA_DIR=str(Path(tmp) / "data"), MODEL_DIR=str(Path(tmp) / "models"), AUTO_RETRAIN_THRESHOLD="3")
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient
from api.main import app

samples = sorted(glob.glob(str(ROOT / "tests" / "samples" / "*.wav")))

def test_flow():
    with TestClient(app) as c:
        assert c.get("/api/health").json()["status"] == "ok"
        r = c.post("/api/predict", files={"file": open(samples[0], "rb")}).json()
        assert r["prediction"] == Path(samples[0]).name.split("_")[0], r
        assert c.post("/api/predict", files={"file": ("x.wav", b"junk")}).status_code == 400
        up = c.post("/api/upload", data={"label": "Dog"}, files=[("files", (Path(p).name, open(p, "rb"))) for p in samples[:1]] * 3).json()
        assert up["accepted"] == 3 and up["auto_retrain_started"], up
        import time; time.sleep(8)
        st = c.get("/api/retrain/status").json()
        assert st["state"] == "done" and st["pending_uploads"] == 0, st
        assert c.get("/api/metrics").json()["current"]["version"] == 2
        assert c.post("/api/upload", data={"label": "x"}, files=[("files", ("a.mp3", b"x"))]).json()["rejected"]
        assert c.get("/").status_code == 200 and c.get("/api/visuals").status_code == 200
    print("all good")

if __name__ == "__main__":
    test_flow()
