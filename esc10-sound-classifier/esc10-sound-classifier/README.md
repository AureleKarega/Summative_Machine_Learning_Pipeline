# ESC-10 Sound Classifier: ML pipeline, API, dashboard and deployment

Classifies 5-second environmental sounds (dog, rooster, rain, sea waves, crackling fire, crying baby, helicopter, chainsaw, clock tick, sneezing). Built on the ESC-10 subset of the public [ESC-50](https://github.com/karolpiczak/ESC-50) dataset.

**Result:** RBF-SVM on 34 audio features, ~82% accuracy / 0.82 macro-F1 on a fixed hold-out fold (random guessing = 10%).

## What is where
| Path | Purpose |
|---|---|
| `notebooks/sound_classifier.ipynb` | Acquisition, processing, visualizations + interpretations, model comparison, full metrics, retraining demo, production-style evaluation |
| `src/features.py`, `src/train.py` | Feature extraction; training, testing, retraining, locking, status |
| `api/main.py` | FastAPI: `/api/predict`, `/api/upload`, `/api/retrain`, `/api/retrain/status`, `/api/health`, `/api/metrics`, `/api/visuals` |
| `ui/index.html` | Dashboard: uptime, metrics, visualizations, predict, bulk upload, retrain button |
| `locustfile.py`, `docker-compose.yml`, `deploy/nginx.conf` | Load testing with N containers behind nginx |
| `Dockerfile`, `render.yaml` | Container image and Render deployment |
| `data/base_features.csv` | Pre-extracted features for the 400 ESC-10 clips (audio is not stored in the repo) |

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python -m src.train                                       # trains models/model.joblib
uvicorn api.main:app --reload                             # dashboard + API at http://localhost:8000
python tests/test_api.py                                  # smoke tests
```
To re-run the notebook from scratch it downloads the ESC-10 audio automatically (~150 MB).

## Retraining
* **Upload:** dashboard → pick a label → select many `.wav` files → *Upload* (features are extracted immediately).
* **Trigger:** (1) *Retrain model now* button / `POST /api/retrain`; (2) automatic when `AUTO_RETRAIN_THRESHOLD` (default 20) new clips are pending; (3) `python -m src.train`.
* Each retrain trains on base data + all uploads, evaluates on the fixed hold-out fold, saves atomically and logs a new version (see history chart).

## Deploy on Render
1. Push this folder to a GitHub repo.
2. Render dashboard → **New → Blueprint** → select the repo (it reads `render.yaml`) → deploy. First build takes a few minutes.
3. Open the URL Render gives you (`https://<name>.onrender.com`). Health check: `/api/health`.

Free-tier notes: the service sleeps after inactivity (first request is slow), has 512 MB RAM, and its disk is **ephemeral**: uploads and retrained models reset on redeploy/restart (the image always starts from the built-in model).

## Load test with Locust and different numbers of containers
```bash
# 1 container
docker compose up --build --scale api=1 -d
locust -f locustfile.py --host http://localhost:8080 --headless -u 100 -r 10 -t 60s --csv results/1_container
docker compose down
# repeat with --scale api=2, then api=4 (use the same -u/-r/-t each time) and --csv results/2_containers, results/4_containers
```
The CSVs in `results/` contain median, 95th percentile, average latency, requests/s and failures. You can also run `locust -f locustfile.py --host <url>` for the live web UI (http://localhost:8089), or point `--host` at the Render URL to test the cloud deployment.

Fill in your own measurements here:

### Load test results

Setup: Locust, 100 concurrent users (spawn rate 10/s), 60 s per run, requests sent through an nginx load balancer to N identical API containers on one Windows machine (Docker Desktop). Each run is a mix of `POST /api/predict` (80%) and `GET /api/health` (20%).

| Containers | Total requests | Failures | Throughput (req/s) | Predict avg (ms) | Predict median (ms) | Predict p95 (ms) |
|---|---|---|---|---|---|---|
| 1 | 429 | 0 | 7.4 | 12,084 | 8,300 | 26,000 |
| 2 | 1,189 | 0 | 20.2 | 4,621 | 4,800 | 6,300 |
| 4 | 1,625 | 0 | 27.1 | 3,283 | 2,900 | 6,600 |

**Observations**
- Throughput rose from 7.4 to 20.2 req/s with 2 containers (about 2.7x) and to 27.1 req/s with 4 containers (about 3.7x). The median prediction time fell from 8.3 s to 2.9 s.
- Scaling is sub-linear. The step from 2 to 4 containers gave about 34% more throughput, because all containers share the same CPU of the host machine. The slowest requests (p95) did not improve between 2 and 4 containers.
- Predictions are CPU-bound (audio feature extraction), so the extra containers help most when requests queue up.
- The 1-container run had no warm-up, so its first predictions were slow (about 23 s) because of a cold start. This makes its latencies look worse than a warmed-up container would show.
- A first 4-container attempt right after startup had 6 failed requests (connection reset) and a long stall before it recovered. The rerun after warm-up completed with 0 failures, which shows how cold starts and overload affect the service.

Sanity check on a laptop-class sandbox (single process, 20 users, 20 s): ~8 req/s, median ≈ 1 s, 0 failures. Feature extraction is CPU-bound, which is why adding containers helps.

## API example
```bash
curl -F "file=@tests/samples/dog_5-203128-A-0.wav" http://localhost:8000/api/predict
```
