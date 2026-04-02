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
2. Add a simple selector baseline such as nearest-centroid or cluster-wise best configuration.
3. After the abstraction feels right, add an ASlib loader.
4. Treat each ASlib solver as either:
   - a direct portfolio member, or
   - a stand-in for a precomputed parameter setting of one base solver

## Sources

- COSEAL ASlib overview: <https://www.coseal.net/aslib/>
- ASlib scenario repository: <https://github.com/coseal/aslib_data>
- OASC 2017 scenario format summary: <https://www.coseal.net/open-algorithm-selection-challenge-2017-oasc/>
