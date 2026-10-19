"""Load test for /predict. Reused for both the PyTorch baseline and the
ONNX Runtime-optimized serving path (the TensorRT substitute — see
src/optimization/export_onnx.py) — point --host at whichever is running.

Generate the two required reports:
    locust -f locustfile.py --host http://localhost:8000 --headless \
        -u 50 -r 10 -t 2m --html reports/locust_fastapi.html

    locust -f locustfile.py --host http://localhost:8002 --headless \
        -u 50 -r 10 -t 2m --html reports/locust_onnx.html
"""

from locust import HttpUser, between, task

SAMPLE_REVIEWS = [
    "المنتج رائع جدا ووصل بسرعة",
    "جودة سيئة جدا ولا انصح بالشراء",
    "التجربة كانت عادية لا بأس بها",
    "خدمة العملاء ممتازة وسريعة الاستجابة",
    "المنتج تالف ولم يطابق الوصف",
]


class SentimentUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task
    def predict(self) -> None:
        text = SAMPLE_REVIEWS[self.environment.runner.user_count % len(SAMPLE_REVIEWS)]
        self.client.post("/predict", json={"text": text})

    @task(1)
    def health(self) -> None:
        self.client.get("/health")
