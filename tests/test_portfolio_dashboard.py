from __future__ import annotations

from isac.analysis import evaluate_dynamic_selectors, evaluate_selectors, make_portfolio_table
from isac.core import DynamicPortfolioBenchmark, PortfolioBenchmark


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
        "Privileged Classifier",
        "MLP Selector",
        "Regressor",
        "Temporal MoE",
    }
    expected_action_rows = 12 + 1 + 12 + 6 * 12
    assert len(action_table) == expected_action_rows
    assert len(instance_table) == 120
    assert "regret_cluster_isac" in instance_table.columns
    assert "regret_dgcac-inspired" in instance_table.columns
    assert "selected_param_1_mlp_selector" in instance_table.columns
    assert "selected_param_1_oracle" in instance_table.columns


def test_make_portfolio_table_matches_number_of_configs() -> None:
    benchmark = PortfolioBenchmark(seed=9)
    portfolio_table = make_portfolio_table(benchmark)

    assert len(portfolio_table) == benchmark.n_configs


def test_evaluate_dynamic_selectors_returns_expected_tables() -> None:
    benchmark = DynamicPortfolioBenchmark(seed=10, horizon=6)
    selector_table, trace_table = evaluate_dynamic_selectors(
        benchmark=benchmark,
        n_episodes=12,
        seed=10,
    )

    assert set(selector_table["selector"]) == {
        "Oracle",
        "Global Best",
        "Random",
        "Cluster ISAC",
        "DGCAC-inspired",
        "Privileged Classifier",
        "MLP Selector",
        "Regressor",
        "Temporal MoE",
        "Temporal Soft Cluster ISAC",
    }
    assert "avg_total_penalty" in selector_table.columns
    assert "avg_switch_cost" in selector_table.columns
    assert "switch_rate" in selector_table.columns
    assert len(trace_table) == 12 * benchmark.horizon * 10
