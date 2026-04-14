# GitHub 上传准备清单

## 已适合上传的内容

- `train.py`：训练入口。
- `evaluate.py`：模型评估入口。
- `rl_utils.py`：环境创建、reward shaping、算法选择等工具函数。
- `configs/`：PPO/SAC 和不同 MuJoCo 环境的配置。
- `scripts/`：常用命令脚本。
- `README.md`：项目说明和运行方式。
- `LEARNING_NOTES.md`：算法、环境、reward shaping 和调参笔记。
- `RESULTS.md`：已验证结果、算力说明、实验路线。
- `PROJECT_REPORT.md`：项目学习过程、架构和结果总结。
- `TRAINING_PLAN.md`：正式训练前的启动说明。
- `.github/workflows/ci.yml`：GitHub 上的轻量语法检查。

## 不应该上传的内容

这些已被 `.gitignore` 排除：

- `.venv/`
- `runs/`
- `.tmp/`
- `__pycache__/`
- `*.zip`
- `*.pkl`
- `*.pt`
- `*.pth`

## 上传前还需要你决定

1. 仓库名，例如 `mujoco-ppo-sac-demo` 或 `rl-mujoco-starter`。
2. 是否公开仓库。
3. 是否添加开源协议。作品集示例项目通常可以选 MIT License。
4. 是否上传训练曲线截图。建议跑完正式实验后再加。

## 推荐第一次提交信息

```text
Initial MuJoCo PPO/SAC training demo
```

## 推荐仓库描述

```text
A reproducible Stable-Baselines3 PPO/SAC training demo on Gymnasium MuJoCo environments.
```
