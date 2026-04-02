# Benchmark Strategy

## Recommendation

Start with a synthetic benchmark that explicitly models:

- a fixed portfolio of pre-selected parameter settings
- observable instance or dataset characteristics
- distinct latent regimes where different parameter settings are optimal

This is the fastest route to the exact decision problem you described:

1. observe features
2. choose one portfolio element
3. measure regret against the best portfolio element for that instance

## Why synthetic first

The main value in the first phase is not realism by itself. It is making sure the repository has the right abstraction:

- portfolio elements are fixed parameter vectors, not policies
- the online decision is selection, not continuous re-optimization
- different subpopulations of instances reward different parameter choices
- features are informative but imperfect

If we start with a heavy real benchmark too early, we risk spending most of our time on parsing, data wrangling, and domain-specific details before the core decision loop is clean.

## Recommended initial simulation

The new `PortfolioBenchmark` in [src/isac/core/portfolio.py](/Users/christianspeck/projects/isac/src/isac/core/portfolio.py) is the recommended starting point.

It works like this:

- each instance belongs to a latent regime
- each regime has a different ideal parameter vector
- the observed features are correlated with the regime
- each portfolio choice is a fixed parameter vector
- runtime increases as the chosen vector moves away from the instance's ideal vector

This gives us a benchmark where:

- one subset of instances prefers a conservative setting
- another prefers an aggressive setting
- another prefers a balanced setting
- some overlap and noise prevent the task from being trivial

That is a close conceptual match to ISAC-style routing.

## Real dataset options for later

The best real-data follow-up is ASlib, which standardizes algorithm-selection scenarios with per-instance features and performance tables across many domains. The official COSEAL ASlib page describes the library and its standard scenario format, and the `aslib_data` repository includes scenarios from SAT, CSP, ASP, MIP, TSP, and OpenML-derived machine learning tasks. Inference: even though ASlib is framed as algorithm selection, it also fits your portfolio-setting view if each "algorithm" is treated as one preselected parameter configuration.

Two especially relevant paths:

- `OPENML-WEKA-2017` for dataset-level meta-features and model-selection style decisions
- SAT or CSP scenarios for classic per-instance routing with strong solver complementarity

## Recommendation order

1. Use the synthetic portfolio benchmark now.
2. Build a dashboard that makes latent regimes, portfolio geometry, selector actions, and regret visible.
3. Compare a small family of selectors under one shared train/test protocol:
   - learned-embedding plus clustering inspired by modern deep ISAC variants
   - clustering with cluster-wise best configuration
   - classification of the best configuration
   - regression of per-configuration runtime
4. After the abstraction feels right, add an ASlib loader.
5. Treat each ASlib solver as either:
   - a direct portfolio member, or
   - a stand-in for a precomputed parameter setting of one base solver

## Paper target for the next selector family

The first modern paper to approximate in this repo is:

- Wen Song, Yi Liu, Zhiguang Cao, Yaoxin Wu, and Qiqiang Li,
  `Instance-specific algorithm configuration via unsupervised deep graph clustering`,
  Engineering Applications of Artificial Intelligence, 125, 2023.

The repo does not yet model graph-structured instances or deep autoencoders directly. The practical first step is a DGCAC-inspired selector that:

1. learns a compact embedding from the current feature vectors,
2. clusters in that latent space,
3. assigns each latent cluster its best portfolio member.

## Robustness evaluation across all three noise sources

The recommended evaluation protocol is a 3D grid sweep:

- `feature_noise in {0.10, 0.40, 0.80}`
- `parameter_noise in {0.02, 0.08, 0.16}`
- `runtime_noise in {0.00, 0.03, 0.08}`

For each grid point:

1. sample a fresh train/test split,
2. fit every selector,
3. record average regret, average runtime, and accuracy,
4. repeat across several seeds,
5. aggregate by mean and worst-case regret.

The repo now includes a scriptable noise sweep and a dashboard heatmap view for exactly this purpose.

## Sources

- COSEAL ASlib overview: <https://www.coseal.net/aslib/>
- ASlib scenario repository: <https://github.com/coseal/aslib_data>
- OASC 2017 scenario format summary: <https://www.coseal.net/open-algorithm-selection-challenge-2017-oasc/>
