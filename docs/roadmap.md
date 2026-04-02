# Roadmap

## Phase 1: Paper-to-simulator bridge

- Keep the environment synthetic and fast.
- Model the runtime decision described in the paper: observe features, pick a configuration, measure regret.
- Include a global fallback baseline that approximates the paper's instance-oblivious configuration.

## Phase 2: Clustering and routing

- Add a clustering module with pluggable strategies.
- Start with `kmeans`, then add automatic cluster count selection.
- Store cluster centers and outlier thresholds for routing new instances.

## Phase 3: Configuration learning

- Replace synthetic configuration quality with trainable surrogate models.
- Add simple cluster-wise optimization loops.
- Track benchmark metrics like average runtime, geometric mean runtime, and slow-down.

## Phase 4: Research environments

- Add budgeted evaluation environments where each configuration trial has a cost.
- Add partial observability and noisy features.
- Add non-stationary instance distributions and transfer settings.

## Phase 5: Real benchmarks

- Introduce loaders for real algorithm-configuration datasets.
- Support reproducible experiment configs and result logging.
