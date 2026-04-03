from __future__ import annotations

from isac.analysis import evaluate_algorithmic_selectors
from isac.core import AlgorithmicPortfolioBenchmark


def test_evaluate_algorithmic_selectors_returns_expected_tables() -> None:
    benchmark = AlgorithmicPortfolioBenchmark(seed=31, horizon=8)
    selector_table, trace_table = evaluate_algorithmic_selectors(
        benchmark=benchmark,
        n_episodes=6,
        seed=31,
        portfolio_source="random_search",
        portfolio_trials=12,
    )

    assert set(selector_table["selector"]) == {
        "Privileged Classifier",
        "Regressor",
        "MLP Selector",
        "DGCAC-inspired",
        "Cluster ISAC",
        "Temporal Soft Cluster ISAC",
    }
    assert "avg_loss" in selector_table.columns
    assert "avg_regret" in selector_table.columns
    assert "portfolio_size" in selector_table.columns
    assert len(trace_table) == 6 * benchmark.horizon * len(selector_table)
