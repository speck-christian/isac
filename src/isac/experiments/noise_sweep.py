"""3D noise-sweep experiments for portfolio selector robustness."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd

from isac.analysis import evaluate_selectors
from isac.core import PortfolioBenchmark


def run_noise_sweep(
    feature_noises: list[float],
    parameter_noises: list[float],
    runtime_noises: list[float],
    *,
    n_instances: int = 240,
    seeds: list[int] | None = None,
    output_csv: str | None = None,
) -> pd.DataFrame:
    """Evaluate all selectors across a grid of benchmark noise settings."""

    seeds = seeds or [0, 1, 2, 3, 4]
    rows: list[dict[str, float | int | str]] = []

    for feature_noise, parameter_noise, runtime_noise in product(
        feature_noises,
        parameter_noises,
        runtime_noises,
    ):
        for seed in seeds:
            benchmark = PortfolioBenchmark(
                feature_noise=feature_noise,
                parameter_noise=parameter_noise,
                runtime_noise=runtime_noise,
                seed=seed,
            )
            selector_table, _, _ = evaluate_selectors(
                benchmark=benchmark,
                n_instances=n_instances,
                seed=seed,
            )
            for record in selector_table.to_dict(orient="records"):
                rows.append(
                    {
                        "feature_noise": feature_noise,
                        "parameter_noise": parameter_noise,
                        "runtime_noise": runtime_noise,
                        "seed": seed,
                        "selector": record["selector"],
                        "avg_runtime": record["avg_runtime"],
                        "avg_regret": record["avg_regret"],
                        "accuracy": record["accuracy"],
                    }
                )

    results = pd.DataFrame(rows)
    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
    return results
