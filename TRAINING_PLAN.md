# 训练计划

这个文件记录我后续怎么继续训练。因为 MuJoCo 训练会占 CPU，我一般会在不用电脑的时候跑正式实验。

## 已完成

- 虚拟环境已创建。
- 依赖已安装。
- `HalfCheetah-v5` 可以正常创建。
- PPO smoke test 已通过。
- PPO + HalfCheetah 200k steps 已完成。
- 模型可以保存和重新评估。

## 已跑主实验

```powershell
python train.py --config configs/ppo_halfcheetah.yaml --total-timesteps 200000 --run-name ppo_halfcheetah_200k_final
```

评估结果：

```text
Mean reward: 102.91 +/- 150.21
```

这个结果我会作为 baseline，不作为最终性能结论。

## 后续训练顺序

我计划按这个顺序继续：

1. PPO + HalfCheetah 跑到 500k。
2. PPO + HalfCheetah 跑 3 个 seed。
3. SAC + HalfCheetah 跑 200k 或 500k。
4. PPO + HalfCheetah reward shaping 对比。
5. 如果 HalfCheetah 结果稳定，再跑 Ant。

## 常用命令

PPO 主实验：

```powershell
.\scripts\train_ppo_halfcheetah.ps1
```

短实验：

```powershell
.\scripts\train_ppo_halfcheetah.ps1 -TotalTimesteps 50000 -RunName ppo_halfcheetah_50k
```

SAC 对比：

```powershell
.\scripts\train_sac_halfcheetah.ps1
```

打开 TensorBoard：

```powershell
.\scripts\open_tensorboard.ps1
```

评估模型：

```powershell
python evaluate.py --model runs\ppo_halfcheetah_200k_final\final_model.zip --episodes 5
```

## 训练时注意

- 插电源。
- 关闭占 CPU 的软件。
- 不要在需要电脑高响应的时候跑长训练。
- 如果电脑明显变卡，可以在终端按 `Ctrl+C` 停止。
