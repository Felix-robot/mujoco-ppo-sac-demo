from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from stable_baselines3.common.evaluation import evaluate_policy

from rl_utils import (
    algo_class,
    build_vec_env,
    load_config,
    load_vec_normalize_if_available,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO/SAC model.")
    parser.add_argument("--model", required=True, help="Path to .zip model file.")
    parser.add_argument("--config", help="Config path. Defaults to config_resolved.yaml near the model.")
    parser.add_argument("--algo", choices=["PPO", "SAC"], help="Override algorithm.")
    parser.add_argument("--env-id", help="Override Gymnasium environment id.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--render", action="store_true", help="Render with render_mode=human.")
    parser.add_argument("--vecnormalize", help="Path to vecnormalize.pkl. Auto-detected when omitted.")
    return parser.parse_args()


def find_run_config(model_path: Path) -> Path | None:
    for parent in [model_path.parent, *model_path.parents]:
        candidate = parent / "config_resolved.yaml"
        if candidate.exists():
            return candidate
    return None


def auto_vecnormalize_path(model_path: Path) -> Path | None:
    for parent in [model_path.parent, *model_path.parents]:
        candidate = parent / "vecnormalize.pkl"
        if candidate.exists():
            return candidate
    return None


def load_eval_config(args: argparse.Namespace) -> dict[str, Any]:
    model_path = Path(args.model)
    config_path = Path(args.config) if args.config else find_run_config(model_path)
    if config_path is None:
        config: dict[str, Any] = {}
    else:
        config = load_config(config_path)

    if args.algo:
        config["algo"] = args.algo
    if args.env_id:
        config["env_id"] = args.env_id
    return config


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    config = load_eval_config(args)

    algo_name = str(config.get("algo", "PPO")).upper()
    env_id = str(config.get("env_id", "HalfCheetah-v5"))
    seed = int(config.get("seed", 42))
    render_mode = "human" if args.render else None

    env = build_vec_env(
        env_id=env_id,
        seed=seed + 20_000,
        n_envs=1,
        env_kwargs=config.get("env_kwargs", {}),
        reward_shaping=config.get("reward_shaping", {}),
        vec_env="dummy",
        normalize={"enabled": False},
        render_mode=render_mode,
    )

    stats_path = Path(args.vecnormalize) if args.vecnormalize else auto_vecnormalize_path(model_path)
    env = load_vec_normalize_if_available(env, stats_path, norm_reward=False)

    model = algo_class(algo_name).load(str(model_path), env=env, device=args.device)
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=args.episodes,
        deterministic=True,
        render=args.render,
    )
    env.close()

    print(f"Algorithm: {algo_name}")
    print(f"Environment: {env_id}")
    print(f"Episodes: {args.episodes}")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main()
