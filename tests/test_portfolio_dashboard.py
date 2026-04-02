from __future__ import annotations

from isac.analysis import evaluate_selectors, make_portfolio_table
from isac.core import PortfolioBenchmark


def test_evaluate_selectors_returns_expected_tables() -> None:
    benchmark = PortfolioBenchmark(seed=8)
    selector_table, action_table, instance_table = evaluate_selectors(
        benchmark=benchmark,
        n_instances=120,
        seed=8,
    )

    assert set(selector_table["selector"]) == {
        "Oracle",
        "Global Best",
        "Random",
        "Cluster ISAC",
        "DGCAC-inspired",
        "Classifier",
        "Regressor",
    }
    assert len(action_table) == benchmark.n_configs * 7
    assert len(instance_table) == 120
    assert "regret_cluster_isac" in instance_table.columns
    assert "regret_dgcac-inspired" in instance_table.columns


def test_make_portfolio_table_matches_number_of_configs() -> None:
    benchmark = PortfolioBenchmark(seed=9)
    portfolio_table = make_portfolio_table(benchmark)

    assert len(portfolio_table) == benchmark.n_configs
