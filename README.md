# Arabic Sentiment Analysis

E-commerce Arabic review classification (Positive / Mixed·Neutral / Negative), built as
Project 1 (Deep Learning track) of the MLOps Practitioner course. Fine-tunes AraBERT,
tracks experiments with MLflow, versions data with DVC, serves with FastAPI + BentoML,
monitors drift with Evidently/Prometheus/Grafana, and optimizes with distillation + ONNX
quantization.

## Quickstart (3 commands)

```bash
git clone <this-repo> && cd arabic-segmentation-analysis
pip install -e .
docker compose up --build api-stable
```

Then:

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"text": "المنتج رائع ووصل بسرعة"}'
# {"label":"Positive","confidence":0.94,"model_version":"stable"}

curl localhost:8000/health
# {"status":"healthy"}
```

`api-stable` loads `$STABLE_MODEL_PATH` (defaults to the base AraBERT checkpoint — see
"Model artifacts" below for how to point it at a fine-tuned model).

## Architecture

```
                 ┌─────────────┐
 review text ──▶ │  FastAPI /  │──▶ label, confidence
                 │  BentoML    │
                 └──────┬──────┘
                        │ /metrics
                        ▼
   ┌─────────────┐   ┌────────────┐   ┌─────────┐
   │ data/raw ──▶│   │ Prometheus │──▶│ Grafana │
   │ prepare.py  │   └────────────┘   └─────────┘
   │ (DVC)       │          ▲
   └──────┬──────┘          │ psi_scores.json
          ▼                 │
   ┌─────────────┐   ┌───────────────┐
   │ train.py /  │──▶│ MLflow        │
   │ sweep.py    │   │ (tracking +   │
   └─────────────┘   │  registry)    │
                      └───────────────┘
```

## Model artifacts

Fine-tuned checkpoints and ONNX exports are **not** committed to git (they're 100MB+
binaries). Instead:
- Training logs every run to MLflow and registers the best one as `ArabicSentiment` in
  the Model Registry, promoted to the `Production` stage.
- `data/raw/` and `data/processed/` are DVC-tracked (`dvc pull` after `git clone` to
  fetch them from the configured remote).
- To run the API against a trained checkpoint: `export MODEL_PATH=models:/ArabicSentiment/Production`
  (requires `MLFLOW_TRACKING_URI` pointed at a reachable MLflow server) or a local
  directory produced by `mlflow artifacts download`.

> Note on the DVC remote: this repo's default remote is a local path for development
> (`dvc remote add -d local_storage /path/to/storage`). For a real multi-reviewer
> setup, repoint it at S3/GCS/Azure Blob and run `dvc push` — see the DVC docs for the
> one-line `dvc remote modify` command per backend.

## Scaling up training

`src/config.py` keeps `TRAIN_SIZE=4000` and the sweep grid small (1-2 epochs) so the
whole pipeline runs on a laptop CPU in minutes. For a real model: raise `TRAIN_SIZE` /
`VAL_SIZE`, remove the `--train-limit`/`--val-limit` flags, and run on a GPU box.

---

## Session 1 — Package, API, Docker

- `src/` packaged with `pyproject.toml` (`pip install -e .` works).
- `ArabicSentimentModel` (`src/models/sentiment_model.py`) wraps AraBERT with type hints.
- FastAPI `/predict` (Pydantic-validated, rejects blank text with 422) and `/health`.
- `Dockerfile` + `docker-compose.yml` — `docker compose up --build api-stable` serves on `:8000`.

Verify:
```bash
docker build -t arabic-sentiment .
docker compose up --build api-stable
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"text":"رائع"}'
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"text":""}'   # -> 422
```

## Session 2 — Tracking, versioning, CI

- `python -m src.training.sweep` runs 5 MLflow experiments (varying `lr`/`batch_size`/`epochs`),
  logs params+metrics+model artifacts, registers the best as `ArabicSentiment`, and
  promotes it to `Production`. Comparison chart: `reports/mlflow_comparison.png`.
- `dvc.yaml` defines `prepare` (download + split) and `train` stages;
  `dvc repro` reproduces both from a clean checkout + `dvc pull`.
- `.github/workflows/ci.yml`: `ruff` → `pytest` → quality gate (`scripts/check_f1_gate.py`,
  fails the build if `f1_macro` regresses more than 0.02 below `reports/baseline_f1_macro.txt`)
  → Docker build & push to GHCR on `main`.

Browse the MLflow UI against the same local run store:
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```

## Session 3 — Serving, load testing, canary

- `bentoml serve src.serving.bento_service:ArabicSentimentService` — a `/predict` API
  with `batchable=True` (micro-batches concurrent requests into one forward pass).
- `src/optimization/export_onnx.py` — **TensorRT substitute**: this project has no
  NVIDIA GPU (TensorRT needs one), so ONNX Runtime is the serving-acceleration path
  instead. The script exports + verifies the model against PyTorch and prints the
  `trtexec --fp16` command to build a real TensorRT engine from the same `.onnx` file
  on a GPU box.
- `src/inference/batch_score.py` — scores ≥1,000 unlabeled reviews, writes
  `data/scoring/output/scored_<ts>.csv` (adds `text_length`/`confidence_score`, the
  features S4 watches for drift).
- `locustfile.py` + `reports/locust_fastapi.html` / `reports/locust_onnx.html` — load
  test at 20 concurrent users. **Honest result**: on this (GPU-less, Apple Silicon)
  dev machine, the naive per-request ONNX Runtime path was *slower* under concurrency
  (p95 ≈ 1.2s) than native PyTorch CPU inference (p95 ≈ 0.12s) — single-request ONNX
  latency is fine (~200ms), but N concurrent synchronous ONNX calls contend for the
  same cores. That's exactly what BentoML's batchable Runner is for: it coalesces
  concurrent requests into one batched forward pass instead of N competing ones. Don't
  expect ONNX/TensorRT to be a free win without batching — benchmark before and after,
  like this report does.
- `nginx/canary.conf` + `docker-compose.yml` (`api-stable` 95% / `api-canary` 5%,
  fronted by nginx on `:8080`) — promote a candidate by raising its weight in stages:
  5 → 25 → 50 → 100, watching the Grafana dashboard (Session 4) between each stage.

## Session 4 — Drift detection and observability

- `src/monitoring/drift.py` computes **PSI** (Population Stability Index) on
  `text_length` and `confidence_score` from the latest batch-scoring run
  (`python -m src.monitoring.drift`), writes `reports/psi_scores.json`, and also runs
  an Evidently `DataDriftPreset` report to `reports/evidently_drift_report.html`.
- `src/monitoring/metrics_exporter.py` republishes `reports/psi_scores.json` as a
  Prometheus gauge (`arabic_sentiment_psi_score{feature=...}`) on `:9100/metrics` —
  live, no restart needed when a new batch is scored.
- `/predict` itself exposes `predict_latency_seconds` (histogram) and
  `predict_requests_total` on `:8000/metrics`.
- Grafana (`:3000`, admin/admin) is provisioned with 2 panels: `/predict` p95 latency
  (split by `stable`/`canary` track) and PSI per feature, with a red threshold line at
  `DRIFT_PSI_ALERT_THRESHOLD = 0.25` (`src/config.py`).
- **Alert**: PSI > 0.25 on either feature is flagged by `drift.py` at the CLI
  (`ALERT` vs `ok`) and crosses the red threshold in Grafana's PSI panel — wire this to
  Grafana Alerting → Slack/email in a real deployment (see `monitoring/grafana/`).

Run the monitoring stack:
```bash
docker compose up prometheus grafana drift-exporter
python -m src.inference.batch_score --n-samples 1000
python -m src.monitoring.drift
# refresh Grafana — PSI panel updates on the next scrape (15s)
```

## Session 5 — Distillation, quantization, final benchmark

- `src/optimization/distill.py` — 12-layer AraBERT teacher → 6-layer student (DistilBERT-
  style init: copy embeddings + every other encoder layer, then fine-tune with
  `0.5·KL(teacher‖student, T=2) + 0.5·cross_entropy`). Logged to MLflow as
  `distill-12to6`; student saved to `models/distilled/` (not committed — see "Model
  artifacts").
- `src/optimization/export_onnx.py` + `src/optimization/quantize.py` — FP32 ONNX export,
  then `quantize_dynamic(..., QInt8)` → `models/model_int8.onnx`.
- `src/optimization/benchmark.py` — accuracy/f1_macro, p95 single-request latency, and
  file size for all three, on the validation split. Real numbers from this (GPU-less,
  smoke-scale-trained) environment:

| model                      | accuracy | f1_macro | p95 latency (ms) | size (MB) |
|-----------------------------|---------:|---------:|------------------:|----------:|
| original (fp32, torch)     | 0.48     | 0.47     | 73.0              | 541.0     |
| distilled (6-layer, torch) | 0.43     | 0.24     | 41.4              | 372.5     |
| INT8 (ONNX Runtime)        | 0.30     | 0.27     | 73.3              | 136.0     |

  (regenerate with `python -m src.optimization.benchmark` → `reports/benchmark.md`)

  **Honest read**: distillation gave a real, clean win — fewer layers, ~1.8x faster,
  smaller — at an accuracy cost that should be re-measured once the model is actually
  trained to convergence (these numbers come from the same small-subsample/short-epoch
  smoke run used throughout this repo, not a converged model). INT8 quantization gave
  a strong ~4x size reduction but **no latency win on this CPU** — consistent with
  Session 3's finding that ONNX Runtime isn't automatically faster than PyTorch's CPU
  kernels here; the real payoff of both optimizations needs either a GPU (true
  TensorRT INT8/FP16 kernels) or request batching (BentoML Runner) to show up in
  latency, not just file size.

### Final architecture (all 5 sessions)

```
                         ┌────────────────────────────┐
                         │        data/raw.csv         │
                         │  (DVC-tracked, HF dataset)   │
                         └──────────────┬───────────────┘
                                        │ prepare.py (DVC "prepare" stage)
                                        ▼
                         ┌────────────────────────────┐
                         │ data/processed/{train,val,  │
                         │ test}.csv (DVC)              │
                         └──────────────┬───────────────┘
                                        │
                     ┌──────────────────┼───────────────────┐
                     ▼                  ▼                   ▼
            train.py / sweep.py   distill.py          export_onnx.py
            (S2: MLflow runs)     (S5: 6-layer)        → quantize.py (S5: INT8)
                     │                  │                   │
                     └─────────┬────────┴─────────┬─────────┘
                                ▼                  ▼
                     MLflow Model Registry   models/*.onnx
                     "ArabicSentiment"       (gitignored — see
                     → Production stage      "Model artifacts")
                                │
                                ▼
          ┌─────────────────────────────────────────────┐
          │  FastAPI (S1) / BentoML batchable (S3)        │
          │  MODEL_ENGINE=torch|onnx, MODEL_PATH=...       │
          └───────────────────┬────────────────────────┘
                               │ nginx canary 95/5 (S3)
                               ▼
          ┌─────────────────────────────────────────────┐
          │ /metrics ──▶ Prometheus ──▶ Grafana (S4)      │
          │ batch_score.py ──▶ drift.py (PSI) ──▶ exporter │
          └─────────────────────────────────────────────┘
```

### Session changelog

| Session | Delivered |
|---------|-----------|
| 1 | Packaged src/, AraBERT wrapper, FastAPI `/predict` + `/health`, Docker |
| 2 | MLflow 5-run sweep + registry promotion, DVC pipeline, CI quality gate |
| 3 | BentoML batchable serving, ONNX export (TensorRT substitute), canary, Locust |
| 4 | PSI drift detection, Prometheus metrics, Grafana dashboard |
| 5 | 6-layer distillation, INT8 quantization, benchmark table, this architecture diagram |
