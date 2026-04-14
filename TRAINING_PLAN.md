# 训练启动准备

现在已经具备开始训练的条件，但不建议在你正在使用电脑时启动正式训练。正式训练会持续占用 CPU，可能让浏览器、IDE 或其他软件变卡。

## 已完成的前置准备

- Python 虚拟环境 `.venv/` 已创建。
- Stable-Baselines3、Gymnasium、MuJoCo、TensorBoard 已安装。
- `HalfCheetah-v5` 已通过环境创建测试。
- PPO smoke test 已跑通。
- 训练、评估、TensorBoard 命令已写成脚本。

## 建议的正式训练

首选实验：

```powershell
.\scripts\train_ppo_halfcheetah.ps1
```

这等价于：

```powershell
python train.py --config configs/ppo_halfcheetah.yaml --total-timesteps 200000 --run-name ppo_halfcheetah_200k
```

如果想先跑更短版本：

```powershell
.\scripts\train_ppo_halfcheetah.ps1 -TotalTimesteps 50000 -RunName ppo_halfcheetah_50k
```

## 对比实验

SAC 对比：

```powershell
.\scripts\train_sac_halfcheetah.ps1
```

Reward shaping 对比：

```powershell
python train.py --config configs/ppo_halfcheetah_shaped.yaml --total-timesteps 200000 --run-name ppo_halfcheetah_shaped_200k
```

## 训练时建议

- 插上电源。
- 关闭大型游戏、视频剪辑、多个 IDE 等重负载软件。
- 训练期间可以继续轻量使用电脑，但不建议同时做需要高响应的工作。
- 如果电脑明显变卡，可以在终端按 `Ctrl+C` 停止训练。

## 训练完成后

打开 TensorBoard：

```powershell
.\scripts\open_tensorboard.ps1
```

评估模型：

```powershell
python evaluate.py --model runs\ppo_halfcheetah_200k\final_model.zip --episodes 5
```

然后把最终 reward 和曲线截图整理到 `RESULTS.md`。
