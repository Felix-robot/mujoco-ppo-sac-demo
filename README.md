# MuJoCo PPO/SAC 强化学习实验

这是我做的一个强化学习入门项目。目标是把一个 RL 训练流程真正跑通，而不是只停留在理论或 notebook 片段上。

我用 Stable-Baselines3 在 Gymnasium MuJoCo 环境里训练 PPO，并保留了 SAC、reward shaping 和调参的配置入口。主实验是：

```text
PPO + HalfCheetah-v5 + 200k steps
```

完整复盘在 [PROJECT_REPORT.md](PROJECT_REPORT.md)，训练结果记录在 [RESULTS.md](RESULTS.md)。

## 我做了什么

- 搭建了一个可以复现的 MuJoCo RL 训练项目。
- 用 PPO 跑通了 `HalfCheetah-v5`。
- 写了独立的训练脚本和评估脚本。
- 用 YAML 管理环境、算法和超参数。
- 加了 reward shaping wrapper，方便后续做对比实验。
- 保存了模型、评估结果和 TensorBoard 日志。
- 把项目整理成可以上传 GitHub 的结构。

## 项目结构

```text
RL/
├── train.py
├── evaluate.py
├── rl_utils.py
├── configs/
├── scripts/
├── README.md
├── PROJECT_REPORT.md
├── RESULTS.md
├── LEARNING_NOTES.md
└── .github/workflows/ci.yml
```

核心文件：

- `train.py`：训练入口，负责读取配置、创建环境、训练模型、保存结果。
- `evaluate.py`：评估入口，负责加载模型并统计平均 reward。
- `rl_utils.py`：环境创建、reward shaping、VecNormalize、算法选择。
- `configs/`：不同算法和环境的配置文件。
- `scripts/`：常用 PowerShell 启动脚本。

## 环境安装

我本地使用的是 Windows + Python 虚拟环境。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

已验证的主要依赖版本：

```text
stable-baselines3 2.8.0
gymnasium 1.2.3
mujoco 3.6.0
torch 2.11.0+cpu
```

## 快速测试

先跑一个很短的 smoke test，确认依赖、MuJoCo 环境、训练和模型保存都正常：

```powershell
python train.py --config configs/ppo_halfcheetah.yaml --smoke-test
```

或者：

```powershell
.\scripts\smoke_test.ps1
```

## 正式训练

主实验命令：

```powershell
python train.py --config configs/ppo_halfcheetah.yaml --total-timesteps 200000 --run-name ppo_halfcheetah_200k
```

或者运行脚本：

```powershell
.\scripts\train_ppo_halfcheetah.ps1
```

SAC 对比实验：

```powershell
python train.py --config configs/sac_halfcheetah.yaml --total-timesteps 200000 --run-name sac_halfcheetah_200k
```

Reward shaping 对比实验：

```powershell
python train.py --config configs/ppo_halfcheetah_shaped.yaml --total-timesteps 200000 --run-name ppo_halfcheetah_shaped_200k
```

## 评估模型

```powershell
python evaluate.py --model runs\ppo_halfcheetah_200k_final\final_model.zip --episodes 5
```

我这次训练后的评估结果：

```text
Mean reward: 102.91 +/- 150.21
```

这个结果说明训练流程已经跑通，但还不是稳定收敛的高性能策略。200k steps 对 MuJoCo 来说偏短，后续还需要更长训练和多 seed 对比。

## 查看训练曲线

```powershell
tensorboard --logdir runs
```

或者：

```powershell
.\scripts\open_tensorboard.ps1
```

## 我目前的结论

这个项目最重要的收获不是 reward 分数，而是完整跑通了 RL 工程流程：

- 配置实验。
- 启动训练。
- 保存模型。
- 加载评估。
- 记录结果。
- 分析局限。

下一步我会优先做三件事：

1. 把 PPO 训练步数提高到 500k 或 1M。
2. 跑 SAC 作为算法对比。
3. 跑 reward shaping 配置，观察学习曲线变化。
