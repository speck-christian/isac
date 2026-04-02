"""Run a short rollout with the random baseline."""

from __future__ import annotations

from isac import make
from isac.baselines import RandomPolicy


def main() -> None:
    env = make("isac-simple-v0", seed=7, horizon=10)
    policy = RandomPolicy(n_actions=env.n_configs, seed=7)

    observation, info = env.reset()
    total_reward = 0.0

    print("Initial info:", info)
    print("First observation shape:", observation.shape)

    terminated = False
    while not terminated:
        action = policy.act(observation)
        observation, reward, terminated, _, step_info = env.step(action)
        total_reward += reward
        print(
            f"action={action} reward={reward:.3f} "
            f"best={step_info['best_config']} regret={step_info['regret']:.3f}"
        )

    print(f"episode_total_reward={total_reward:.3f}")


if __name__ == "__main__":
    main()
