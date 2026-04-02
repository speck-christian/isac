# ISAC Sim

`isac-sim` is a research-first Python package for building simulation environments inspired by the ISAC paper, "ISAC: Instance-Specific Algorithm Configuration," by Kadioglu, Malitsky, Sellmann, and Tierney.

The initial repo goal is intentionally modest:

- start from the core paper idea that instance features can guide algorithm configuration
- expose that idea through a simple simulation environment
- keep the repository easy to extend toward richer clustering, tuning, and learned policies

The original paper clusters normalized instance features, learns a configuration per cluster, and falls back to a global configuration for instances far from all known clusters. This repository starts with a synthetic version of that interaction loop so new models can be added incrementally.

Paper:
- ResearchGate PDF: <https://www.researchgate.net/profile/Yuri-Malitsky/publication/220837402_ISAC_-_Instance-Specific_Algorithm_Configuration/links/02e7e52738cb135ccc000000/ISAC-Instance-Specific-Algorithm-Configuration.pdf>

## Repository layout

```text
src/isac/
  baselines/      Simple policies and reference strategies
  core/           Synthetic data generation and shared utilities
  envs/           Simulation environments
tests/            Unit tests
examples/         Minimal runnable scripts
docs/             Notes and roadmap
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python examples/run_random_policy.py
./scripts/build_report.sh
```

## First environment

The first environment, `isac-simple-v0`, generates synthetic problem instances with feature vectors and asks an agent to choose one of several algorithm configurations. Reward is the negative regret relative to the best available configuration for that instance.

This gives us a compact experimental loop for:

- routing instances to configurations
- evaluating cluster-aware heuristics
- comparing policies against a global fallback baseline

## Recommended benchmark direction

The clearest next benchmark is a synthetic portfolio-selection setup where each selectable action is a fixed parameter vector and different subsets of instances prefer different vectors. That keeps the focus on online selection from a limited portfolio, which matches your intended problem framing better than continuous control.

See [docs/benchmark_strategy.md](/Users/christianspeck/projects/isac/docs/benchmark_strategy.md) for the recommended synthetic benchmark and the planned path to real ASlib scenarios later.

## Technical report

A textbook-style living report is maintained in [docs/report/technical_report.tex](/Users/christianspeck/projects/isac/docs/report/technical_report.tex), with generated output at `docs/report/build/technical_report.pdf`.

## Roadmap

- add explicit clustering modules that mirror the paper more closely
- add offline datasets and benchmark loaders
- add richer environments with budgeted tuning and delayed feedback
- support both heuristic and learned policies

See [docs/roadmap.md](/Users/christianspeck/projects/isac/docs/roadmap.md) for the staged plan.
