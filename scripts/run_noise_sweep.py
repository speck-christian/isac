"""CLI entrypoint for the 3D noise-sweep experiment."""

from __future__ import annotations

import argparse

from isac.experiments import run_noise_sweep


def parse_float_list(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a 3D noise sweep over the portfolio benchmark."
    )
    parser.add_argument("--feature-noises", default="0.1,0.4,0.8")
    parser.add_argument("--parameter-noises", default="0.02,0.08,0.16")
    parser.add_argument("--runtime-noises", default="0.0,0.03,0.08")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--instances", type=int, default=240)
    parser.add_argument("--output", default="artifacts/noise_sweep/results.csv")
    args = parser.parse_args()

    results = run_noise_sweep(
        feature_noises=parse_float_list(args.feature_noises),
        parameter_noises=parse_float_list(args.parameter_noises),
        runtime_noises=parse_float_list(args.runtime_noises),
        n_instances=args.instances,
        seeds=parse_int_list(args.seeds),
        output_csv=args.output,
    )
    print(f"wrote {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
