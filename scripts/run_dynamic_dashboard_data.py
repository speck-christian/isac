"""Precompute the main dynamic dashboard artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isac.analysis import evaluate_dynamic_selectors
from isac.core import DynamicPortfolioBenchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute the main dynamic dashboard artifacts.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--drift-scale", type=float, default=0.22)
    parser.add_argument("--regime-switch-prob", type=float, default=0.18)
    parser.add_argument("--switching-cost", type=float, default=0.04)
    parser.add_argument("--observation-noise", type=float, default=0.10)
    parser.add_argument("--missing-feature-prob", type=float, default=0.22)
    parser.add_argument("--multimodal-surface-scale", type=float, default=0.30)
    parser.add_argument("--artifact-tag", default="default")
    parser.add_argument(
        "--summary-output",
        default="artifacts/dynamic/dashboard_selector_summary.csv",
    )
    parser.add_argument("--trace-output", default="artifacts/dynamic/dashboard_trace_table.csv")
    parser.add_argument("--metadata-output", default="artifacts/dynamic/dashboard_metadata.json")
    args = parser.parse_args()

    benchmark = DynamicPortfolioBenchmark(
        horizon=args.horizon,
        drift_scale=args.drift_scale,
        regime_switch_prob=args.regime_switch_prob,
        switching_cost=args.switching_cost,
        observation_noise=args.observation_noise,
        missing_feature_prob=args.missing_feature_prob,
        multimodal_surface_scale=args.multimodal_surface_scale,
        seed=args.seed,
    )
    selector_table, trace_table = evaluate_dynamic_selectors(
        benchmark=benchmark,
        n_episodes=args.episodes,
        seed=args.seed,
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
                "drift_scale": args.drift_scale,
                "regime_switch_prob": args.regime_switch_prob,
                "switching_cost": args.switching_cost,
                "observation_noise": args.observation_noise,
                "missing_feature_prob": args.missing_feature_prob,
                "multimodal_surface_scale": args.multimodal_surface_scale,
            },
            indent=2,
        )
    )

    print(f"wrote selector summary to {summary_output}")
    print(f"wrote trace table to {trace_output}")
    print(f"wrote metadata to {metadata_output}")


if __name__ == "__main__":
    main()
