"""Prometheus exporter: reads reports/psi_scores.json (written by src.monitoring.drift)
and republishes it as a gauge on every scrape, so Grafana's PSI panel stays live.

Run as: python -m src.monitoring.metrics_exporter
"""

from __future__ import annotations

import json
import time

from prometheus_client import REGISTRY, start_http_server
from prometheus_client.core import GaugeMetricFamily

from src.config import DRIFT_PSI_ALERT_THRESHOLD, REPORTS_DIR

PSI_FILE = REPORTS_DIR / "psi_scores.json"
PORT = 9100


class PSICollector:
    def collect(self):
        gauge = GaugeMetricFamily(
            "arabic_sentiment_psi_score",
            "Population Stability Index per feature",
            labels=["feature"],
        )
        if PSI_FILE.exists():
            scores = json.loads(PSI_FILE.read_text())
            for feature, psi in scores.items():
                gauge.add_metric([feature], psi)
        yield gauge

        alert = GaugeMetricFamily(
            "arabic_sentiment_psi_alert_threshold", "Configured PSI alert threshold"
        )
        alert.add_metric([], DRIFT_PSI_ALERT_THRESHOLD)
        yield alert


if __name__ == "__main__":
    REGISTRY.register(PSICollector())
    start_http_server(PORT)
    print(f"Serving PSI metrics on :{PORT}/metrics")
    while True:
        time.sleep(3600)
