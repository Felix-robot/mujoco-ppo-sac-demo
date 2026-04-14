# 实验结果与算力说明

## 当前已验证结果

本项目已经在 Windows 本地环境完成了一次最小 smoke test：

```powershell
.\.venv\Scripts\python.exe train.py --config configs/ppo_halfcheetah.yaml --smoke-test --run-name smoke_ppo_halfcheetah
```

验证内容：

- Stable-Baselines3 可以正常导入。
- Gymnasium MuJoCo `HalfCheetah-v5` 可以正常创建和 step。
- PPO 可以完成短训练。
- 模型可以保存、恢复并评估。

环境信息：

```text
stable-baselines3 2.8.0
gymnasium 1.2.3
mujoco 3.6.0
torch 2.11.0+cpu
```

HalfCheetah 空间信息：

```text
observation shape: (17,)
action shape: (6,)
```

Smoke test 评估结果：

```text
Algorithm: PPO
Environment: HalfCheetah-v5
Episodes: 2
Mean reward: -1.25 +/- 0.29
```

这个结果只说明训练链路跑通。由于 smoke test 只训练 1024 steps，reward 不代表最终性能。

## 正式训练结果

主实验：

```text
Algorithm: PPO
Environment: HalfCheetah-v5
Requested timesteps: 200000
Actual timesteps: 204800
Run directory: runs/ppo_halfcheetah_200k_final
Device: CPU
```

训练完成后，`final_model.zip` 已生成，并使用 `evaluate.py` 评估 5 个 episode：

```text
Algorithm: PPO
Environment: HalfCheetah-v5
Episodes: 5
Mean reward: 102.91 +/- 150.21
```

同时评估 `best_model/best_model.zip`：

```text
Algorithm: PPO
Environment: HalfCheetah-v5
Episodes: 5
Mean reward: 41.35 +/- 78.15
```

训练期间的 EvalCallback 在 200000 steps 处记录：

```text
Mean reward: 47.75 +/- 87.42
```

结果说明：这次训练已经完成完整工程闭环，但 200k steps 对 MuJoCo 连续控制任务仍然偏短，评估方差较大。该结果适合作为 GitHub 示例项目的首个 baseline，后续可以通过更长训练、多 seed、SAC 对比和 reward shaping 进一步增强。

## 算力预估

MuJoCo 连续控制任务主要消耗 CPU 环境步进，神经网络更新可以用 GPU 加速，但小型 MLP 在 CPU 上也能跑。

推荐目标：

| 环境 | 推荐算法 | 适合作为首次实验 | 本地 CPU 可行性 | 备注 |
| --- | --- | --- | --- | --- |
| `HalfCheetah-v5` | PPO/SAC | 是 | 高 | 最适合先跑通和展示 |
| `Ant-v5` | PPO/SAC | 可以 | 中 | 训练更慢，效果更有展示性 |
| `Humanoid-v5` | PPO | 不建议首次 | 低到中 | 维度高，通常需要更长训练 |

粗略建议：

- `HalfCheetah-v5 + PPO + 50k steps`：可以作为快速本地实验。
- `HalfCheetah-v5 + PPO + 200k steps`：适合作为 GitHub 示例的正式结果。
- `Ant-v5 + PPO + 500k steps`：适合作为进阶结果。
- `Humanoid-v5 + PPO + 1M steps`：除非有较多时间或云服务器，否则不建议作为主结果。

## 是否需要云服务器

不一定需要。

如果目标是做一个 GitHub 示例项目，本地 CPU 跑 `HalfCheetah-v5` 就够了。重点是代码结构、可复现命令、实验记录和对 reward shaping/hyperparameter tuning 的理解。

建议使用云服务器的情况：

- 想跑 `Humanoid-v5`。
- 想做多组 seed 对比。
- 想系统性调参，例如每个参数跑 3 个 seed。
- 希望在短时间内得到更好看的曲线和演示模型。

不建议一开始就上云。先本地跑通 `HalfCheetah-v5 + PPO`，确认项目完整，再决定是否扩展。

## 下一步实验路线

1. 跑正式 PPO：

```powershell
python train.py --config configs/ppo_halfcheetah.yaml --total-timesteps 200000
```

2. 跑 SAC 对比：

```powershell
python train.py --config configs/sac_halfcheetah.yaml --total-timesteps 200000
```

3. 跑 reward shaping 对比：

```powershell
python train.py --config configs/ppo_halfcheetah_shaped.yaml --total-timesteps 200000
```

4. 打开 TensorBoard：

```powershell
tensorboard --logdir runs
```

5. 评估最终模型：

```powershell
python evaluate.py --model runs\你的实验目录\final_model.zip --episodes 5
```

## GitHub 展示建议

建议上传：

- 训练与评估代码。
- YAML 配置。
- `README.md`。
- `LEARNING_NOTES.md`。
- `RESULTS.md`。

不建议上传：

- `.venv/`
- `runs/`
- `.tmp/`
- `__pycache__/`
- 大模型文件或 TensorBoard 原始日志。

如果要展示训练结果，可以只上传一张 TensorBoard 曲线截图，或者在 `RESULTS.md` 里记录最终 mean reward。
