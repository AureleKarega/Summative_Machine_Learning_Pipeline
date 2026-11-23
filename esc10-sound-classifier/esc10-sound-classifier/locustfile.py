"""Flood the model with requests.
UI mode : locust -f locustfile.py --host http://localhost:8080
Headless: locust -f locustfile.py --host http://localhost:8080 --headless -u 100 -r 10 -t 60s --csv results/run_1c
"""
import glob, random
from locust import HttpUser, task, between

SAMPLES = [open(p, "rb").read() for p in sorted(glob.glob("tests/samples/*.wav"))]


class SoundUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(8)
    def predict(self):
        data = random.choice(SAMPLES)
        self.client.post("/api/predict", files={"file": ("clip.wav", data, "audio/wav")}, name="POST /api/predict")

    @task(1)
    def health(self):
        self.client.get("/api/health", name="GET /api/health")
