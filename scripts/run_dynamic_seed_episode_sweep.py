"""Precompute the dynamic dashboard seed/episode robustness sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from isac.analysis import evaluate_dynamic_selectors
from isac.core import DynamicPortfolioBenchmark


def parse_int_list(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute the dynamic dashboard seed/episode sweep artifact."
    )
    parser.add_argument("--episodes", default="20,40,80,120,160,200")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--drift-scale", type=float, default=0.22)
    parser.add_argument("--regime-switch-prob", type=float, default=0.18)
    parser.add_argument("--switching-cost", type=float, default=0.04)
    parser.add_argument("--observation-noise", type=float, default=0.10)
    parser.add_argument("--missing-feature-prob", type=float, default=0.22)
    parser.add_argument("--multimodal-surface-scale", type=float, default=0.30)
    parser.add_argument("--output", default="artifacts/dynamic/seed_episode_sweep.csv")
    args = parser.parse_args()

    rows: list[dict[str, float | int | str]] = []
    for seed in parse_int_list(args.seeds):
        for n_episodes in parse_int_list(args.episodes):
            benchmark = DynamicPortfolioBenchmark(
                horizon=args.horizon,
                drift_scale=args.drift_scale,
                regime_switch_prob=args.regime_switch_prob,
                switching_cost=args.switching_cost,
                observation_noise=args.observation_noise,
                missing_feature_prob=args.missing_feature_prob,
                multimodal_surface_scale=args.multimodal_surface_scale,
                seed=seed,
            )
            selector_table, _ = evaluate_dynamic_selectors(
                benchmark=benchmark,
                n_episodes=n_episodes,
                seed=seed,
            )
            for record in selector_table.to_dict(orient="records"):
                rows.append(
                    {
                        "seed": seed,
                        "episodes": n_episodes,
                        "selector": record["selector"],
                        "avg_total_penalty": record["avg_total_penalty"],
                        "avg_regret": record["avg_regret"],
                        "avg_switch_cost": record["avg_switch_cost"],
                    }
                )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
