"""Drift detection on batch-scoring output: PSI on text_length/confidence_score,
plus a full Evidently DataDrift report. Run after every batch scoring run.

Run as: python -m src.monitoring.drift --reference <csv> --current <csv>
"""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.config import DRIFT_PSI_ALERT_THRESHOLD, REPORTS_DIR, SCORING_DIR

PSI_FEATURES = ["text_length", "confidence_score"]


def compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index between two numeric distributions."""
    breakpoints = np.quantile(reference, np.linspace(0, 1, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    breakpoints = np.unique(breakpoints)

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    ref_pct = np.clip(ref_counts / max(len(reference), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(len(current), 1), 1e-6, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def latest_scored_batch() -> pd.DataFrame:
    files = sorted(glob.glob(str(SCORING_DIR / "output" / "scored_*.csv")))
    if not files:
        raise FileNotFoundError("No scored batches found — run src.inference.batch_score first")
    return pd.read_csv(files[-1])


def run_drift_check(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, float]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    psi_scores = {
        feature: round(compute_psi(reference[feature], current[feature]), 4)
        for feature in PSI_FEATURES
    }
    (REPORTS_DIR / "psi_scores.json").write_text(json.dumps(psi_scores, indent=2))

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(
        current_data=current[PSI_FEATURES], reference_data=reference[PSI_FEATURES]
    )
    snapshot.save_html(str(REPORTS_DIR / "evidently_drift_report.html"))

    for feature, psi in psi_scores.items():
        flag = "ALERT" if psi > DRIFT_PSI_ALERT_THRESHOLD else "ok"
        print(f"PSI[{feature}] = {psi:.4f} ({flag}, threshold={DRIFT_PSI_ALERT_THRESHOLD})")

    return psi_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=str, default=None)
    parser.add_argument("--current", type=str, default=None)
    args = parser.parse_args()

    current_df = pd.read_csv(args.current) if args.current else latest_scored_batch()
    reference_df = (
        pd.read_csv(args.reference)
        if args.reference
        else current_df.sample(frac=0.5, random_state=0)
    )
    run_drift_check(reference_df, current_df)
