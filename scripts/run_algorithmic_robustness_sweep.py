"""Precompute a multi-seed algorithmic robustness sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from isac.analysis import evaluate_algorithmic_selectors
from isac.core import AlgorithmicPortfolioBenchmark


def parse_int_list(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute a multi-seed algorithmic robustness sweep artifact."
    )
    parser.add_argument("--seeds", default="31,32,33")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--observation-noise", type=float, default=0.08)
    parser.add_argument("--regime-switch-prob", type=float, default=0.14)
    parser.add_argument("--portfolio-source", choices=["random_search", "smac"], default="smac")
    parser.add_argument("--portfolio-trials", type=int, default=16)
    parser.add_argument("--artifact-tag", default="robustness_default")
    parser.add_argument("--output", default="artifacts/algorithmic/robustness_sweep.csv")
    parser.add_argument(
        "--metadata-output",
        default="artifacts/algorithmic/robustness_metadata.json",
    )
    args = parser.parse_args()

    rows: list[dict[str, float | int | str]] = []
    for seed in parse_int_list(args.seeds):
        benchmark = AlgorithmicPortfolioBenchmark(
            horizon=args.horizon,
            observation_noise=args.observation_noise,
            regime_switch_prob=args.regime_switch_prob,
            seed=seed,
        )
        selector_table, _trace_table = evaluate_algorithmic_selectors(
            benchmark=benchmark,
            n_episodes=args.episodes,
            seed=seed,
            portfolio_source=args.portfolio_source,
            portfolio_trials=args.portfolio_trials,
        )
        for record in selector_table.to_dict(orient="records"):
            rows.append(
                {
                    "seed": seed,
                    "selector": record["selector"],
                    "portfolio_source": record["portfolio_source"],
                    "portfolio_size": record["portfolio_size"],
                    "avg_loss": record["avg_loss"],
                    "avg_regret": record["avg_regret"],
                }
            )

    output_path = Path(args.output)
    metadata_output_path = Path(args.metadata_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(output_path, index=False)
    metadata_output_path.write_text(
        json.dumps(
            {
                "artifact_tag": args.artifact_tag,
                "seeds": parse_int_list(args.seeds),
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

    print(f"wrote {len(rows)} rows to {output_path}")
    print(f"wrote metadata to {metadata_output_path}")


if __name__ == "__main__":
    main()
