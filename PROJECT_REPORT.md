# MuJoCo PPO/SAC 强化学习示例项目报告

## 项目概述

本项目是一个基于 Stable-Baselines3、Gymnasium 和 MuJoCo 的强化学习训练示例。项目目标不是追求最高分数，而是完整实现一个可复现的 RL 实验流程：安装环境、创建 MuJoCo 任务、训练 PPO/SAC、保存模型、评估结果、记录实验配置，并为后续 reward shaping 和 hyperparameter tuning 留出清晰入口。

本次正式训练选择 `HalfCheetah-v5 + PPO` 作为主实验。原因是 HalfCheetah 是连续控制任务中较轻量的 MuJoCo 环境，适合作为首次跑通和 GitHub 展示项目；PPO 稳定、工程复杂度适中，适合作为基线算法。

## 我学到的基础知识

### 强化学习训练闭环

强化学习任务的核心是 agent 与 environment 的循环交互：

1. 环境返回 observation。
2. 策略网络根据 observation 输出 action。
3. 环境执行 action，返回 next observation、reward 和 episode 是否结束。
4. 算法根据收集到的轨迹更新策略。
5. 重复这个过程，直到策略在任务上获得更高回报。

在这个项目里，MuJoCo 环境负责物理模拟，Stable-Baselines3 负责 PPO/SAC 的算法实现，训练脚本负责把配置、环境、算法、日志和模型保存串起来。

### PPO 的基本思想

PPO 是一种 on-policy 策略梯度算法。它使用当前策略采样数据，并通过 clipping 限制策略更新幅度，减少一次更新过大导致训练崩溃的风险。

在本项目中，PPO 的关键超参数包括：

- `learning_rate`：控制每次梯度更新幅度。
- `n_steps`：每个环境采样多少步再更新。
- `batch_size`：每次梯度更新使用多少样本。
- `n_epochs`：同一批 rollout 数据被重复训练多少轮。
- `gamma`：未来奖励折扣因子。
- `gae_lambda`：优势估计的偏差和方差折中。

### SAC 的基本思想

SAC 是 off-policy 算法，会把经验存入 replay buffer 并重复利用。它还引入 entropy maximization，让策略在追求 reward 的同时保持探索能力。SAC 通常样本效率更高，但对 replay buffer、warmup steps 和 entropy 系数更敏感。

项目中保留了 `configs/sac_halfcheetah.yaml`，可以直接作为 PPO 的对比实验。

### MuJoCo 连续控制环境

MuJoCo 任务通常是连续动作空间，不是离散动作选择。以 `HalfCheetah-v5` 为例：

- observation shape: `(17,)`
- action shape: `(6,)`

这类任务关注机器人控制策略，例如如何通过连续力矩让 HalfCheetah 向前移动。相比 CartPole 这类入门环境，MuJoCo 更接近真实控制问题，也更适合作为 RL 工程能力展示。

### Reward shaping

Reward shaping 是在环境原始 reward 之外添加额外奖励或惩罚，引导 agent 更快学到目标行为。例如在 HalfCheetah 中，可以额外鼓励 `reward_forward`，也可以加强对 `reward_ctrl` 的控制代价约束。

本项目没有直接修改 Gymnasium 环境源码，而是在 `RewardShapingWrapper` 中实现可配置 reward shaping。这样可以通过 YAML 切换实验，保持原始环境和 shaped reward 实验可对比。

### Hyperparameter tuning

调参不是随机改参数，而是控制变量实验。合理流程是：

1. 固定环境和算法，先跑通 baseline。
2. 固定总步数，对比不同 seed 的波动。
3. 对比 PPO 和 SAC。
4. 只修改一个或少数超参数，例如 `learning_rate` 或 `batch_size`。
5. 用 TensorBoard 曲线和最终评估 reward 判断效果。

本项目通过 YAML 配置保存所有关键参数，便于复现实验和记录调参过程。

## 项目的基本架构

```text
RL/
├── train.py                         # 训练入口：读取配置、构建环境、创建算法、保存模型
├── evaluate.py                      # 评估入口：加载模型并统计平均 reward
├── rl_utils.py                      # 环境构建、reward shaping、VecNormalize、算法选择
├── requirements.txt                 # Python 依赖
├── configs/
│   ├── ppo_halfcheetah.yaml          # PPO + HalfCheetah baseline
│   ├── sac_halfcheetah.yaml          # SAC + HalfCheetah 对比实验
│   ├── ppo_halfcheetah_shaped.yaml   # Reward shaping 对比实验
│   ├── ppo_ant.yaml                  # Ant 扩展环境
│   └── ppo_humanoid.yaml             # Humanoid 扩展环境
├── scripts/
│   ├── setup_env.ps1                 # 创建虚拟环境并安装依赖
│   ├── smoke_test.ps1                # 快速验证训练链路
│   ├── train_ppo_halfcheetah.ps1     # 启动 PPO 主实验
│   ├── train_sac_halfcheetah.ps1     # 启动 SAC 对比实验
│   └── open_tensorboard.ps1          # 打开 TensorBoard
├── README.md                         # 项目运行说明
├── RESULTS.md                        # 实验结果和算力说明
├── TRAINING_PLAN.md                  # 正式训练启动清单
├── LEARNING_NOTES.md                 # 学习笔记
├── GITHUB_PREP.md                    # GitHub 上传准备清单
└── .github/workflows/ci.yml          # GitHub Actions 语法检查
```

### 训练流程

`train.py` 的核心流程：

1. 从 YAML 读取实验配置。
2. 根据 `env_id` 创建 Gymnasium MuJoCo 环境。
3. 可选启用 reward shaping 和 VecNormalize。
4. 根据 `algo` 创建 PPO 或 SAC。
5. 启动训练，并定期运行 `EvalCallback`。
6. 保存 checkpoint、best model、final model 和归一化统计。

### 评估流程

`evaluate.py` 的核心流程：

1. 加载训练 run 目录中的 `config_resolved.yaml`。
2. 重建同样的 MuJoCo 环境。
3. 如果存在 `vecnormalize.pkl`，加载训练时的归一化统计。
4. 加载模型并运行多个 episode。
5. 输出 mean reward 和 standard deviation。

### 配置设计

项目把环境、算法和超参数放在 YAML 中，而不是硬编码在 Python 文件里。这使得实验可以通过命令行快速切换：

```powershell
python train.py --config configs/ppo_halfcheetah.yaml --total-timesteps 200000
python train.py --config configs/sac_halfcheetah.yaml --total-timesteps 200000
python train.py --config configs/ppo_halfcheetah_shaped.yaml --total-timesteps 200000
```

这种结构更适合作为示例项目，因为代码负责实验框架，配置负责实验变量。

## 完成之后的结果

### 环境与依赖

本地环境：

```text
stable-baselines3 2.8.0
gymnasium 1.2.3
mujoco 3.6.0
torch 2.11.0+cpu
```

训练设备：CPU

主实验：

```text
Algorithm: PPO
Environment: HalfCheetah-v5
Requested timesteps: 200000
Actual timesteps: 204800
Run directory: runs/ppo_halfcheetah_200k_final
```

PPO 实际执行到 `204800` steps，是因为 PPO 按 rollout batch 更新，最终步数会对齐到 `n_envs * n_steps` 的倍数。

### Smoke test

在正式训练前，先运行了 1024 steps 的 smoke test，验证依赖、环境、训练、保存和评估链路可用：

```text
Algorithm: PPO
Environment: HalfCheetah-v5
Episodes: 2
Mean reward: -1.25 +/- 0.29
```

该结果只用于证明工程链路跑通，不代表模型性能。

### 正式训练结果

训练过程中，`EvalCallback` 每 10000 steps 评估一次。部分评估点如下：

| Timesteps | Mean reward |
| --- | ---: |
| 10000 | -0.37 |
| 50000 | -2.86 |
| 100000 | -18.41 |
| 150000 | -18.08 |
| 190000 | -25.83 |
| 200000 | 47.75 |

训练结束后，使用 `evaluate.py` 对 final model 评估 5 个 episode：

```text
Algorithm: PPO
Environment: HalfCheetah-v5
Episodes: 5
Mean reward: 102.91 +/- 150.21
```

同时评估 `best_model` checkpoint：

```text
Algorithm: PPO
Environment: HalfCheetah-v5
Episodes: 5
Mean reward: 41.35 +/- 78.15
```

### 结果解读

这次训练证明了完整 RL 工程流程已经跑通：

- MuJoCo 环境可以正常创建和交互。
- PPO 可以完成训练并保存模型。
- 训练日志、checkpoint、best model、final model 都能生成。
- 模型可以被重新加载并独立评估。
- TensorBoard 可以用于观察训练曲线。

从结果上看，200k steps 已经让模型从初始随机策略进入可学习状态，但还不是稳定高性能策略。最终评估标准差较大，说明策略在不同 episode 上表现波动明显。这是合理现象，因为 200k steps 对 MuJoCo 连续控制任务仍然偏短。

更严格的下一步应该是：

1. 将 PPO 训练步数提升到 500k 或 1M。
2. 至少跑 3 个不同 seed，报告均值和方差。
3. 跑 `SAC + HalfCheetah` 作为算法对比。
4. 跑 `ppo_halfcheetah_shaped.yaml`，分析 reward shaping 是否改善学习速度或稳定性。
5. 在结果稳定后扩展到 `Ant-v5`。

## 项目展示价值

这个项目可以展示以下能力：

- 能搭建可复现的 RL 实验工程，而不是只运行单个 notebook。
- 能使用 Stable-Baselines3 接入 MuJoCo 连续控制任务。
- 能通过配置管理算法、环境和超参数。
- 能实现可切换的 reward shaping，而不破坏原始环境。
- 能保存、恢复、评估模型，并解释结果的局限性。
- 能把实验过程整理成适合 GitHub 展示的项目结构。

因此，这个项目适合作为强化学习入门阶段的工程示例，也可以作为后续算法对比、调参和更复杂 MuJoCo 任务的基础。
