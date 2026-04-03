from __future__ import annotations

import numpy as np

from isac.core import DynamicPortfolioEpisode, DynamicPortfolioState
from isac.selectors import TemporalMixtureOfExpertsSelector


def test_temporal_moe_predicts_episode_actions() -> None:
    selector = TemporalMixtureOfExpertsSelector(
        n_configs=2,
        max_portfolio_size=3,
        n_experts=2,
        epochs=60,
        seed=0,
    )
    episode = DynamicPortfolioEpisode(
        states=[
            DynamicPortfolioState(
                features=np.array([0.0, 0.1, 0.0]),
                latent_features=np.array([0.0, 0.1, 0.0]),
                observation_mask=np.ones(3),
                ideal_params=np.array([0.10, 0.90]),
                base_difficulty=1.0,
                runtimes=np.array([1.0, 2.0]),
                best_config=0,
                regime_id=0,
                timestep=0,
            ),
            DynamicPortfolioState(
                features=np.array([0.1, 0.1, 0.1]),
                latent_features=np.array([0.1, 0.1, 0.1]),
                observation_mask=np.ones(3),
                ideal_params=np.array([0.14, 0.86]),
                base_difficulty=1.0,
                runtimes=np.array([1.1, 1.9]),
                best_config=0,
                regime_id=0,
                timestep=1,
            ),
            DynamicPortfolioState(
                features=np.array([3.0, 3.1, 2.9]),
                latent_features=np.array([3.0, 3.1, 2.9]),
                observation_mask=np.ones(3),
                ideal_params=np.array([0.86, 0.24]),
                base_difficulty=1.0,
                runtimes=np.array([2.1, 1.1]),
                best_config=1,
                regime_id=1,
                timestep=2,
            ),
        ]
    )

    selector.fit_dynamic([episode])
    episode_features = np.stack([state.features for state in episode.states], axis=0)
    actions = selector.predict_episode(episode_features, switching_cost=0.04)

    assert actions.shape == (3,)
    assert np.all(actions >= 0)
    assert np.all(actions < selector.portfolio_values_.shape[0])
