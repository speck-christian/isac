"""Precompute the main algorithmic dashboard artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isac.analysis import evaluate_algorithmic_selectors
from isac.core import AlgorithmicPortfolioBenchmark


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute the main algorithmic dashboard artifacts."
    )
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--observation-noise", type=float, default=0.08)
    parser.add_argument("--regime-switch-prob", type=float, default=0.14)
    parser.add_argument("--portfolio-source", choices=["random_search", "smac"], default="smac")
    parser.add_argument("--portfolio-trials", type=int, default=12)
    parser.add_argument("--artifact-tag", default="default")
    parser.add_argument(
        "--summary-output",
        default="artifacts/algorithmic/dashboard_selector_summary.csv",
    )
    parser.add_argument("--trace-output", default="artifacts/algorithmic/dashboard_trace_table.csv")
    parser.add_argument(
        "--metadata-output",
        default="artifacts/algorithmic/dashboard_metadata.json",
    )
    args = parser.parse_args()

    benchmark = AlgorithmicPortfolioBenchmark(
        horizon=args.horizon,
        observation_noise=args.observation_noise,
        regime_switch_prob=args.regime_switch_prob,
        seed=args.seed,
    )
    selector_table, trace_table = evaluate_algorithmic_selectors(
        benchmark=benchmark,
        n_episodes=args.episodes,
        seed=args.seed,
        portfolio_source=args.portfolio_source,
        portfolio_trials=args.portfolio_trials,
    )

    summary_output = Path(args.summary_output)
    trace_output = Path(args.trace_output)
    metadata_output = Path(args.metadata_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    trace_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)

    selector_table.to_csv(summary_output, index=False)
    trace_table.to_csv(trace_output, index=False)
    metadata_output.write_text(
        json.dumps(
            {
                "artifact_tag": args.artifact_tag,
                "seed": args.seed,
                "n_episodes": args.episodes,
                "horizon": args.horizon,
                "observation_noise": args.observation_noise,
                "regime_switch_prob": args.regime_switch_prob,
                "portfolio_source": args.portfolio_source,
                "portfolio_trials": args.portfolio_trials,
            },
            indent=2,
        )
    )

    print(f"wrote selector summary to {summary_output}")
    print(f"wrote trace table to {trace_output}")
    print(f"wrote metadata to {metadata_output}")


if __name__ == "__main__":
    main()
