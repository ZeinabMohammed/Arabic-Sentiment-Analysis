import numpy as np
import pandas as pd

from src.monitoring.drift import compute_psi


def test_psi_is_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(0)
    data = pd.Series(rng.normal(50, 10, 500))
    assert compute_psi(data, data) == 0.0


def test_psi_is_large_for_shifted_distributions() -> None:
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(50, 10, 500))
    current = pd.Series(rng.normal(150, 10, 500))
    assert compute_psi(reference, current) > 0.25
