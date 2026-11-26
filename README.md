# ESC-10 Sound Classifier: ML pipeline, API, dashboard and cloud deployment

## 1. Video demo
**YouTube:** (https://youtu.be/q-fvm4c1-GA)

## 2. Live URL
**Deployed app (Render):** https://summative-machine-learning-pipeline-90rc.onrender.com

The free Render instance sleeps after inactivity, so the first request can take up to ~50 seconds to wake it. Health check: `/api/health`. Interactive API docs: `/docs`.

## 3. Project description
This project classifies **5-second environmental sound clips** into 10 classes (dog, rooster, rain, sea waves, crackling fire, crying baby, helicopter, chainsaw, clock tick, sneezing) using the **ESC-10** subset of the public [ESC-50](https://github.com/karolpiczak/ESC-50) dataset (400 clips, 40 per class, 5 predefined folds).

It covers the full ML lifecycle:

| Stage | Where |
|---|---|
| Data acquisition | `notebooks/sound_classifier.ipynb` section 1, `scripts/build_features.py` |
| Data processing | notebook section 2 (audit, standardisation, 34 audio features, validation, scaling), `src/features.py` |
| Model creation | notebook sections 4-5 (Logistic Regression vs Random Forest vs RBF-SVM), `src/train.py` |
| Model testing / prediction | notebook sections 5-5b (metrics + `predict_file` / `predict_batch` functions), `/api/predict` |
| Retraining and trigger | `src/train.py` (`run_retrain_job`), button + `POST /api/retrain`, automatic trigger at 20 uploaded clips |
| API | `api/main.py` (FastAPI) |
| UI | `ui/index.html` (uptime, visualizations with interpretations, predict, bulk upload, retrain button) |
| Deployment | `Dockerfile`, `render.yaml` (Render), `docker-compose.yml` + nginx (multi-container) |
| Load testing | `locustfile.py`, results below |

**Model:** StandardScaler + RBF-SVM on 34 features (13 MFCC means, 13 MFCC stds, spectral centroid/bandwidth/roll-off/flatness, zero-crossing rate, RMS energy).
**Evaluation:** trained on folds 1-4, tested on the never-trained fold 5 (80 clips). Accuracy **82.5%**, macro precision 0.845, macro recall 0.825, macro F1 **0.822**, log loss 0.728, ROC-AUC (one-vs-rest) 0.978. Random guessing would be 10%.

**Model file:** `models/model.pkl` (a Python pickle containing the scikit-learn pipeline, the feature order and a version number). Load it with:
```python
import pickle
bundle = pickle.load(open("models/model.pkl", "rb"))
bundle["pipeline"].predict_proba(features_df[bundle["features"]])
```

### Dashboard functionality
1. **Predict** one `.wav` clip: class, confidence and top-3 probabilities.
2. **Visualizations** with written interpretations of three features: brightness (spectral centroid), burstiness (RMS std) and energy (RMS mean).
3. **Bulk upload** of labelled `.wav` files for retraining (a new label creates a new class).
4. **Retrain** button (also triggers automatically when 20 new clips are pending); every retrain creates a new model version with its metrics.
5. **Monitoring:** model uptime, model version, metrics per version, predictions served.

## 4. Setup
Requirements: Python 3.10+ and Git. Docker Desktop is only needed for the multi-container test.

### Run locally
```bash
git clone https://github.com/AureleKarega/Summative_Machine_Learning_Pipeline.git
cd Summative_Machine_Learning_Pipeline/esc10-sound-classifier/esc10-sound-classifier   # folder that contains the Dockerfile
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements-dev.txt
python -m src.train               # trains and writes models/model.pkl
uvicorn api.main:app --reload     # open http://localhost:8000
```
Run the checks with `python tests/test_api.py`.

### Run the notebook
```bash
jupyter notebook notebooks/sound_classifier.ipynb
```
On first run it downloads the ESC-10 audio (~150 MB) into `ESC-50/`. It works on a temporary copy, so it never overwrites the repo's model. To evaluate the *deployed* model, set `BASE_URL` before launching (`$env:BASE_URL="https://summative-machine-learning-pipeline-90rc.onrender.com"` on PowerShell) and re-run section 7.

### Run with Docker (1 or more containers behind nginx)
```bash
docker compose up --build --scale api=3      # dashboard at http://localhost:8080
docker compose down
```

### Deploy on Render
1. Push the repo to GitHub.
2. Render → **New → Web Service** → connect the repo.
3. Language **Docker**; set **Root Directory** to the folder that contains the `Dockerfile` (here `esc10-sound-classifier/esc10-sound-classifier`); instance type Free; health check path `/api/health`.
4. Deploy. Note: on the free tier the disk is ephemeral, so uploaded clips and retrained models reset when the service restarts.

### Retraining
Dashboard → type a label → choose several `.wav` files → **Upload files** → **Retrain model now**. Or call `POST /api/upload` (form fields `label`, `files`) and `POST /api/retrain`.

## 5. Results from the flood request simulation
Tool: [Locust](https://locust.io). Setup: 100 concurrent users (spawn rate 10/s), 60 s per run, requests sent through an nginx load balancer to N identical API containers on one Windows laptop (Docker Desktop). Each run is a mix of `POST /api/predict` (80%, a 5 s `.wav` clip) and `GET /api/health` (20%). Raw Locust CSVs are in the [`results/` folder](https://github.com/AureleKarega/Summative_Machine_Learning_Pipeline/tree/main/esc10-sound-classifier/esc10-sound-classifier/results).

```bash
locust -f locustfile.py --host http://localhost:8080 --headless -u 100 -r 10 -t 60s --csv results/N_containers
```

| Containers | Total requests | Failures | Throughput (req/s) | Predict avg (ms) | Predict median (ms) | Predict p95 (ms) |
|---|---|---|---|---|---|---|
| 1 (cold start) | 429 | 0 | 7.4 | 12,084 | 8,300 | 26,000 |
| 2 | 1,189 | 0 | 20.2 | 4,621 | 4,800 | 6,300 |
| 3 | 1,911 | 0 | 32.3 | 2,737 | 2,400 | 6,200 |
| 4 | 1,625 | 0 | 27.1 | 3,283 | 2,900 | 6,600 |

**Observations**
- Going from 1 to 3 containers raised throughput from 7.4 to 32.3 requests/s (about 4x) and cut the median prediction time from 8.3 s to 2.4 s.
- Throughput peaked at 3 containers; 4 containers was slightly lower. All containers share the host's CPU, so more containers stop helping once the CPU is saturated. Warm-up also differed between runs (none for 1 container, 20 s for 2 and 4, 40 s for 3), so part of the 3-vs-4 gap may be run-to-run variation.
- Prediction is CPU-bound (audio feature extraction), which is why queuing dominates latency with few containers.
- The 1-container run had no warm-up, so its latency includes cold-start slowness (the first predictions took about 23 s).
- A first 4-container attempt right after startup had 6 failed requests out of 1,009 (connection reset, 0.6%) and a long stall. The rerun after warm-up had 0 failures.
- Under light load (8 users, 3 warmed-up containers) the median prediction time was about 180 ms, so the multi-second latencies above come from 100 users queuing for the CPU, not from slow individual predictions.

## Repository layout
```
api/main.py            FastAPI service
src/features.py        audio feature extraction (shared by training and API)
src/train.py           training, evaluation, retraining, locking and status
ui/index.html          dashboard
notebooks/             sound_classifier.ipynb (preprocessing, training, testing, prediction functions)
models/model.pkl       trained model file (pickle)
data/base_features.csv pre-extracted features for the 400 ESC-10 clips
results/               Locust load-test CSVs
locustfile.py, docker-compose.yml, deploy/nginx.conf, Dockerfile, render.yaml
tests/                 smoke tests and sample .wav clips
```
