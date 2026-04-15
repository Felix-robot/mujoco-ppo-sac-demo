# 实验结果记录

这个文件记录我目前已经跑过的实验结果。训练产物没有上传到 GitHub，模型和 TensorBoard 日志都保存在本地 `runs/` 目录。

## 环境

```text
OS: Windows
Device: CPU
stable-baselines3: 2.8.0
gymnasium: 1.2.3
mujoco: 3.6.0
torch: 2.11.0+cpu
```

`HalfCheetah-v5` 空间信息：

```text
observation shape: (17,)
action shape: (6,)
```

## Smoke test

命令：

```powershell
python train.py --config configs/ppo_halfcheetah.yaml --smoke-test --run-name smoke_ppo_halfcheetah
```

结果：

```text
Algorithm: PPO
Environment: HalfCheetah-v5
Episodes: 2
Mean reward: -1.25 +/- 0.29
```

这个测试只训练了 1024 steps。我用它确认环境、依赖、训练、保存和评估都能正常运行。

## 正式训练：PPO + HalfCheetah

训练配置：

```text
Algorithm: PPO
Environment: HalfCheetah-v5
Requested timesteps: 200000
Actual timesteps: 204800
Run directory: runs/ppo_halfcheetah_200k_final
Device: CPU
```

训练命令：

```powershell
python train.py --config configs/ppo_halfcheetah.yaml --total-timesteps 200000 --run-name ppo_halfcheetah_200k_final
```

训练过程中的评估记录：

| Timesteps | Mean reward | Std |
| --- | ---: | ---: |
| 10000 | -0.37 | 0.91 |
| 20000 | -0.26 | 0.46 |
| 50000 | -2.86 | 0.51 |
| 100000 | -18.41 | 6.28 |
| 150000 | -18.08 | 1.25 |
| 190000 | -25.83 | 0.57 |
| 200000 | 47.75 | 87.42 |

训练结束后评估 final model：

```text
Algorithm: PPO
Environment: HalfCheetah-v5
Episodes: 5
Mean reward: 102.91 +/- 150.21
```

评估 best checkpoint：

```text
Algorithm: PPO
Environment: HalfCheetah-v5
Episodes: 5
Mean reward: 41.35 +/- 78.15
```

## 结果分析

这次训练完成了我最开始设定的目标：项目能训练、能保存模型、能重新加载模型评估。

结果还不稳定。`final_model` 的平均 reward 比 smoke test 高，但标准差很大，说明策略还没有稳定学会 HalfCheetah 的动作模式。200k steps 对 MuJoCo 连续控制任务来说比较短。

所以我把这次结果定位为 baseline。它能证明工程流程是通的，但还不能证明算法效果已经很好。

## 算力感受

这次是在 CPU 上跑完的。`HalfCheetah-v5 + PPO + 200k steps` 本地可以接受，训练时间不算夸张。

我的判断：

- HalfCheetah 可以本地跑。
- Ant 可以本地试，但会更慢。
- Humanoid 不适合作为第一阶段目标。
- 如果要多 seed 或大量调参，再考虑云服务器。

## 下一步实验

我后续计划：

1. PPO 跑到 500k 或 1M steps。
2. 跑 SAC 作为对比。
3. 跑 reward shaping 配置。
4. 每组实验至少用 3 个 seed。
5. 把 TensorBoard 曲线截图补进报告。
