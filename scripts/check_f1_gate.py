"""CI quality gate: fail the build if f1_macro regresses below the committed baseline.

Reads reports/train_metrics.json (written by src.training.train --metrics-out)
and compares it against reports/baseline_f1_macro.txt. Update the baseline file
deliberately whenever a new model is promoted to Production.
"""

import json
import sys
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
TOLERANCE = 0.02


def main() -> int:
    metrics = json.loads((REPORTS_DIR / "train_metrics.json").read_text())
    baseline = float((REPORTS_DIR / "baseline_f1_macro.txt").read_text().strip())

    current = metrics["f1_macro"]
    print(f"baseline f1_macro={baseline:.4f} current f1_macro={current:.4f}")

    if current < baseline - TOLERANCE:
        print(f"FAIL: f1_macro dropped more than {TOLERANCE} below baseline")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
