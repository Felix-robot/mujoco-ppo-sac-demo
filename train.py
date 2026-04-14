from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from rl_utils import algo_class, build_vec_env, deep_update, load_config, save_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO/SAC on Gymnasium MuJoCo tasks.")
    parser.add_argument("--config", default="configs/ppo_halfcheetah.yaml", help="YAML config path.")
    parser.add_argument("--algo", choices=["PPO", "SAC"], help="Override algorithm.")
    parser.add_argument("--env-id", help="Override Gymnasium environment id.")
    parser.add_argument("--total-timesteps", type=int, help="Override training timesteps.")
    parser.add_argument("--seed", type=int, help="Override random seed.")
    parser.add_argument("--n-envs", type=int, help="Override number of vectorized environments.")
    parser.add_argument("--device", help="cpu, cuda, or auto.")
    parser.add_argument("--run-name", help="Output run name under runs/.")
    parser.add_argument("--eval-freq", type=int, help="Evaluate every N environment timesteps.")
    parser.add_argument("--save-freq", type=int, help="Checkpoint every N environment timesteps.")
    parser.add_argument("--reward-scale", type=float, help="Multiply environment reward by this value.")
    parser.add_argument("--no-normalize", action="store_true", help="Disable VecNormalize.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a tiny training job to verify setup.")
    return parser.parse_args()


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updates: dict[str, Any] = {
        "algo": args.algo,
        "env_id": args.env_id,
        "total_timesteps": args.total_timesteps,
        "seed": args.seed,
        "n_envs": args.n_envs,
        "device": args.device,
        "run_name": args.run_name,
        "eval": {"freq": args.eval_freq},
        "save": {"freq": args.save_freq},
        "reward_shaping": {"reward_scale": args.reward_scale},
    }

    if args.no_normalize:
        updates["normalize"] = {"enabled": False}

    config = deep_update(config, updates)

    if args.smoke_test:
        config["total_timesteps"] = min(int(config.get("total_timesteps", 1024)), 1024)
        config["n_envs"] = 1
        config["vec_env"] = "dummy"
        config.setdefault("eval", {})["freq"] = 512
        config.setdefault("eval", {})["episodes"] = 1
        config.setdefault("save", {})["freq"] = 512

        hyperparams = config.setdefault("hyperparams", {})
        if str(config.get("algo", "PPO")).upper() == "PPO":
            hyperparams["n_steps"] = min(int(hyperparams.get("n_steps", 64)), 64)
            hyperparams["batch_size"] = min(int(hyperparams.get("batch_size", 64)), 64)
            hyperparams["n_epochs"] = min(int(hyperparams.get("n_epochs", 1)), 1)
        elif str(config.get("algo", "SAC")).upper() == "SAC":
            hyperparams["learning_starts"] = min(int(hyperparams.get("learning_starts", 10)), 10)
            hyperparams["buffer_size"] = min(int(hyperparams.get("buffer_size", 1000)), 1000)
            hyperparams["batch_size"] = min(int(hyperparams.get("batch_size", 64)), 64)

    return config


def make_run_dir(config: dict[str, Any]) -> Path:
    run_root = Path(config.get("run_root", "runs"))
    if config.get("run_name"):
        run_name = str(config["run_name"])
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{config['env_id']}_{str(config['algo']).upper()}_seed{config['seed']}_{timestamp}"
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)

    algo_name = str(config.get("algo", "PPO")).upper()
    env_id = str(config.get("env_id", "HalfCheetah-v5"))
    seed = int(config.get("seed", 42))
    n_envs = int(config.get("n_envs", 1))
    total_timesteps = int(config.get("total_timesteps", 100_000))

    run_dir = make_run_dir(config)
    save_config(config, run_dir / "config_resolved.yaml")

    train_env = build_vec_env(
        env_id=env_id,
        seed=seed,
        n_envs=n_envs,
        env_kwargs=config.get("env_kwargs", {}),
        reward_shaping=config.get("reward_shaping", {}),
        vec_env=str(config.get("vec_env", "dummy")),
        normalize=config.get("normalize", {}),
    )
    eval_env = build_vec_env(
        env_id=env_id,
        seed=seed + 10_000,
        n_envs=1,
        env_kwargs=config.get("env_kwargs", {}),
        reward_shaping=config.get("reward_shaping", {}),
        vec_env="dummy",
        normalize=config.get("normalize", {}),
    )

    callback_freq_scale = max(n_envs, 1)
    eval_freq = max(int(config.get("eval", {}).get("freq", 10_000)) // callback_freq_scale, 1)
    save_freq = max(int(config.get("save", {}).get("freq", 50_000)) // callback_freq_scale, 1)

    callbacks = CallbackList(
        [
            EvalCallback(
                eval_env,
                best_model_save_path=str(run_dir / "best_model"),
                log_path=str(run_dir / "eval"),
                eval_freq=eval_freq,
                n_eval_episodes=int(config.get("eval", {}).get("episodes", 5)),
                deterministic=True,
                render=False,
            ),
            CheckpointCallback(
                save_freq=save_freq,
                save_path=str(run_dir / "checkpoints"),
                name_prefix=f"{algo_name.lower()}_{env_id}",
            ),
        ]
    )

    model_cls = algo_class(algo_name)
    model = model_cls(
        policy=str(config.get("policy", "MlpPolicy")),
        env=train_env,
        seed=seed,
        verbose=int(config.get("verbose", 1)),
        tensorboard_log=str(run_dir / "tb"),
        device=str(config.get("device", "auto")),
        **dict(config.get("hyperparams", {})),
    )

    print(f"Training {algo_name} on {env_id}")
    print(f"Run directory: {run_dir}")
    print(f"Total timesteps: {total_timesteps}")

    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
        model.save(run_dir / "final_model")
        if config.get("normalize", {}).get("enabled", False):
            train_env.save(str(run_dir / "vecnormalize.pkl"))
        print(f"Saved final model to {run_dir / 'final_model.zip'}")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
