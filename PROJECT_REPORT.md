# 项目复盘：MuJoCo PPO/SAC 强化学习实验

## 1. 我为什么做这个项目

我做这个项目是为了把强化学习从“看懂概念”推进到“能实际跑起来”。

一开始我对 PPO、SAC、MuJoCo、reward shaping、调参这些词有概念，但没有把它们连成一个完整工程。所以我给自己定了一个明确目标：用 Stable-Baselines3 跑通一个 MuJoCo 连续控制任务，并把训练、评估、结果记录和项目结构都整理好。

我最后选的是：

```text
PPO + HalfCheetah-v5
```

原因很直接：HalfCheetah 比 Ant 和 Humanoid 轻，适合第一次跑通；PPO 比较稳定，适合作为 baseline。

## 2. 我学到的基础知识

### 强化学习训练闭环

这次项目让我把 RL 的基本闭环真正跑了一遍：

1. 环境给出 observation。
2. 策略网络根据 observation 输出 action。
3. MuJoCo 执行动作并返回 reward。
4. PPO 根据收集到的数据更新策略。
5. 训练脚本定期评估并保存模型。

之前我容易把 RL 理解成“调一个算法”，实际做完以后发现，工程上更重要的是把环境、算法、配置、日志、评估和模型文件管理清楚。

### PPO

PPO 是 on-policy 算法。它用当前策略采样数据，然后用 clipping 限制策略更新幅度，避免每次更新变化太大。

这次我重点理解了几个参数：

- `learning_rate`：更新步子有多大。
- `n_steps`：每轮采样多少步。
- `batch_size`：每次训练用多少样本。
- `n_epochs`：同一批数据重复训练几轮。
- `gamma`：未来 reward 的折扣。
- `gae_lambda`：优势估计的平滑程度。

我目前的理解是：PPO 的优势是稳定，适合先做 baseline；缺点是样本效率不算高，需要比较多交互步数。

### SAC

SAC 是 off-policy 算法，会把经验放进 replay buffer 重复利用，同时用 entropy 鼓励探索。

我这次没有把 SAC 作为主结果，但项目里保留了 `configs/sac_halfcheetah.yaml`。后续可以直接用同一个训练框架跑 SAC，对比 PPO 的学习速度和稳定性。

### MuJoCo 连续控制

这次用的是 `HalfCheetah-v5`：

```text
observation shape: (17,)
action shape: (6,)
```

这和 CartPole 这种离散动作环境不同。HalfCheetah 的动作是连续力矩，模型要学的是怎样控制身体向前移动。MuJoCo 的任务更接近机器人控制问题，也更适合作为 RL 项目展示。

### Reward shaping

Reward shaping 是在原始 reward 基础上加一些额外奖励或惩罚，让 agent 更容易学到目标行为。

我没有直接改 Gymnasium 的环境源码，而是写了一个 `RewardShapingWrapper`。这样做的好处是：

- 原始环境保持不变。
- shaping 参数可以写在 YAML 里。
- 后续对比实验更清楚。

例如 `configs/ppo_halfcheetah_shaped.yaml` 里可以对 `reward_forward` 和 `reward_ctrl` 重新加权。

### Hyperparameter tuning

这次我也更清楚了调参应该怎么做。不是随便改参数，而是控制变量：

1. 先固定环境和算法，跑一个 baseline。
2. 再换 seed，看结果波动。
3. 再对比 PPO 和 SAC。
4. 再改 learning rate、batch size、n_steps 这类关键参数。
5. 每次只改少数变量，用 TensorBoard 曲线看变化。

## 3. 项目的基本架构

项目结构如下：

```text
RL/
├── train.py
├── evaluate.py
├── rl_utils.py
├── configs/
│   ├── ppo_halfcheetah.yaml
│   ├── sac_halfcheetah.yaml
│   ├── ppo_halfcheetah_shaped.yaml
│   ├── ppo_ant.yaml
│   └── ppo_humanoid.yaml
├── scripts/
│   ├── setup_env.ps1
│   ├── smoke_test.ps1
│   ├── train_ppo_halfcheetah.ps1
│   ├── train_sac_halfcheetah.ps1
│   └── open_tensorboard.ps1
├── README.md
├── RESULTS.md
├── LEARNING_NOTES.md
└── .github/workflows/ci.yml
```

### `train.py`

`train.py` 是训练入口。它做几件事：

- 读取 YAML 配置。
- 创建 MuJoCo 环境。
- 根据配置选择 PPO 或 SAC。
- 启动训练。
- 定期评估。
- 保存 checkpoint、best model、final model。

### `evaluate.py`

`evaluate.py` 负责加载训练好的模型并跑多个 episode，最后输出平均 reward 和标准差。

如果训练时用了 VecNormalize，评估脚本会加载对应的 `vecnormalize.pkl`，避免训练和评估环境不一致。

### `rl_utils.py`

这个文件放公共逻辑：

- 创建单个环境。
- 创建 vectorized env。
- reward shaping wrapper。
- 算法名称到 SB3 类的映射。
- VecNormalize 的加载逻辑。

我把这些逻辑拆出来，是为了让 `train.py` 和 `evaluate.py` 保持简单。

### `configs/`

我把实验参数放进 YAML，而不是写死在 Python 里。这样后续想换环境、算法、步数或超参数，不需要改代码。

目前已有配置：

- PPO + HalfCheetah
- SAC + HalfCheetah
- PPO + HalfCheetah reward shaping
- PPO + Ant
- PPO + Humanoid

## 4. 完成之后的结果

### Smoke test

我先跑了一个 1024 steps 的 smoke test，确认训练链路可用：

```text
Algorithm: PPO
Environment: HalfCheetah-v5
Episodes: 2
Mean reward: -1.25 +/- 0.29
```

这个结果不代表性能，只说明依赖、环境、训练、保存和评估都能跑通。

### 正式训练

正式训练设置：

```text
Algorithm: PPO
Environment: HalfCheetah-v5
Requested timesteps: 200000
Actual timesteps: 204800
Device: CPU
Run directory: runs/ppo_halfcheetah_200k_final
```

PPO 实际跑到 `204800` steps，是因为它按 rollout batch 更新，最终步数会对齐到 batch 大小。

训练过程中部分评估点：

| Timesteps | Mean reward |
| --- | ---: |
| 10000 | -0.37 |
| 50000 | -2.86 |
| 100000 | -18.41 |
| 150000 | -18.08 |
| 190000 | -25.83 |
| 200000 | 47.75 |

训练完成后，我用 `evaluate.py` 评估 final model：

```text
Episodes: 5
Mean reward: 102.91 +/- 150.21
```

也评估了 best checkpoint：

```text
Episodes: 5
Mean reward: 41.35 +/- 78.15
```

### 我的判断

这次结果说明项目已经完成了基本目标：RL 训练工程完整跑通，模型可以保存和重新评估。

但模型还没有稳定收敛。最终 reward 的标准差很大，说明策略在不同 episode 上表现波动明显。200k steps 对 MuJoCo 来说偏短，所以这个结果更适合作为 baseline，而不是最终性能展示。

## 5. 下一步计划

我后面会按这个顺序继续做：

1. 把 PPO 训练步数提高到 500k 或 1M。
2. 跑 3 个不同 seed，观察结果稳定性。
3. 跑 SAC，对比 PPO。
4. 跑 reward shaping 配置，看学习速度有没有变化。
5. 如果 HalfCheetah 结果稳定，再扩展到 Ant。

## 6. 这个项目能展示什么

这个项目可以展示我已经掌握了一个基础 RL 工程流程：

- 能搭建可复现的训练项目。
- 能使用 Stable-Baselines3 和 Gymnasium MuJoCo。
- 能用配置文件管理实验。
- 能保存、加载和评估模型。
- 能理解 reward shaping 和调参的基本思路。
- 能对结果做基本分析，而不是只贴一个分数。
