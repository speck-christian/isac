from __future__ import annotations

from isac.experiments import run_noise_sweep


def test_noise_sweep_returns_expected_grid_rows() -> None:
    results = run_noise_sweep(
        feature_noises=[0.1, 0.4],
        parameter_noises=[0.02],
        runtime_noises=[0.0, 0.03],
        n_instances=60,
        seeds=[0, 1],
    )

    expected_selectors = {
        "Oracle",
        "Global Best",
        "Random",
        "Cluster ISAC",
        "DGCAC-inspired",
        "Classifier",
        "MLP Selector",
        "Regressor",
    }
    assert set(results["selector"]) == expected_selectors
    assert len(results) == 2 * 1 * 2 * 2 * len(expected_selectors)
