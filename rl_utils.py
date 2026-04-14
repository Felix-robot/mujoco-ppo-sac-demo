from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import gymnasium as gym
import yaml
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize


ALGORITHMS = {
    "PPO": PPO,
    "SAC": SAC,
}


class RewardShapingWrapper(gym.Wrapper):
    """Apply light, configurable reward shaping without editing MuJoCo env code."""

    def __init__(
        self,
        env: gym.Env,
        reward_scale: float = 1.0,
        extra_info_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__(env)
        self.reward_scale = reward_scale
        self.extra_info_weights = extra_info_weights or {}

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = float(reward) * self.reward_scale
        shaping_terms: dict[str, float] = {}

        for info_key, weight in self.extra_info_weights.items():
            raw_value = float(info.get(info_key, 0.0))
            contribution = weight * raw_value
            shaped_reward += contribution
            shaping_terms[f"shape/{info_key}"] = contribution

        info = dict(info)
        info["reward/original"] = float(reward)
        info["reward/shaped"] = float(shaped_reward)
        info.update(shaping_terms)
        return obs, shaped_reward, terminated, truncated, info


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in updates.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def algo_class(name: str):
    normalized = name.upper()
    if normalized not in ALGORITHMS:
        choices = ", ".join(sorted(ALGORITHMS))
        raise ValueError(f"Unsupported algorithm '{name}'. Choose one of: {choices}")
    return ALGORITHMS[normalized]


def make_single_env(
    env_id: str,
    seed: int,
    rank: int = 0,
    env_kwargs: dict[str, Any] | None = None,
    reward_shaping: dict[str, Any] | None = None,
    render_mode: str | None = None,
):
    env_kwargs = dict(env_kwargs or {})
    reward_shaping = reward_shaping or {}

    if render_mode is not None:
        env_kwargs["render_mode"] = render_mode

    def _init():
        env = gym.make(env_id, **env_kwargs)
        env = RewardShapingWrapper(
            env,
            reward_scale=float(reward_shaping.get("reward_scale", 1.0)),
            extra_info_weights=reward_shaping.get("extra_info_weights", {}),
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env

    return _init


def build_vec_env(
    env_id: str,
    seed: int,
    n_envs: int,
    env_kwargs: dict[str, Any] | None = None,
    reward_shaping: dict[str, Any] | None = None,
    vec_env: str = "dummy",
    normalize: dict[str, Any] | None = None,
    render_mode: str | None = None,
) -> VecEnv:
    env_fns = [
        make_single_env(
            env_id=env_id,
            seed=seed,
            rank=rank,
            env_kwargs=env_kwargs,
            reward_shaping=reward_shaping,
            render_mode=render_mode,
        )
        for rank in range(n_envs)
    ]

    if vec_env == "subproc" and n_envs > 1:
        env: VecEnv = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)

    normalize = normalize or {}
    if normalize.get("enabled", False):
        env = VecNormalize(
            env,
            norm_obs=bool(normalize.get("norm_obs", True)),
            norm_reward=bool(normalize.get("norm_reward", True)),
            clip_obs=float(normalize.get("clip_obs", 10.0)),
            gamma=float(normalize.get("gamma", 0.99)),
        )
    return env


def load_vec_normalize_if_available(
    env: VecEnv,
    stats_path: str | Path | None,
    norm_reward: bool = False,
) -> VecEnv:
    if stats_path is None:
        return env
    stats_path = Path(stats_path)
    if not stats_path.exists():
        return env
    loaded = VecNormalize.load(str(stats_path), env)
    loaded.training = False
    loaded.norm_reward = norm_reward
    return loaded
