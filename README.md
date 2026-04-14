# MuJoCo RL PPO/SAC 实验

这个文件夹实现了一个最小但完整的强化学习任务：用 Stable-Baselines3 在 Gymnasium MuJoCo 环境上训练 PPO 或 SAC，并保留 reward shaping 与 hyperparameter tuning 的实验入口。

项目学习过程、架构和正式训练结果见 `PROJECT_REPORT.md`。

## 任务描述是什么意思

“跑通一个 RL 算法”不是只写伪代码，而是要真正完成一次训练闭环：

1. 选一个连续控制环境，例如 `HalfCheetah-v5`、`Ant-v5` 或 `Humanoid-v5`。
2. 选一个算法，例如 `PPO` 或 `SAC`。
3. 安装 MuJoCo、Gymnasium、Stable-Baselines3。
4. 启动训练，保存模型、日志、评估结果。
5. 能解释 reward shaping 和 hyperparameter tuning 对训练结果的影响。

这里默认推荐先跑 `HalfCheetah-v5 + PPO`，因为它比 `Ant` 和 `Humanoid` 更轻，适合先验证整条流程。

## 你需要先做的事

如果当前 Python 3.13 安装依赖失败，建议装 Python 3.11 或 3.12，再建虚拟环境。Windows PowerShell 示例：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

如果你确定当前 Python 能装这些包，也可以直接：

```powershell
python -m pip install -r requirements.txt
```

## 快速烟雾测试

先用很短的训练步数验证 MuJoCo 和依赖是否能跑：

```powershell
python train.py --config configs/ppo_halfcheetah.yaml --smoke-test
```

成功后会在 `runs/` 下生成一次实验目录，里面有模型、配置、TensorBoard 日志和评估记录。

也可以运行脚本：

```powershell
.\scripts\smoke_test.ps1
```

## 正式训练

PPO + HalfCheetah：

```powershell
python train.py --config configs/ppo_halfcheetah.yaml
```

Windows PowerShell 脚本：

```powershell
.\scripts\train_ppo_halfcheetah.ps1
```

SAC + HalfCheetah：

```powershell
python train.py --config configs/sac_halfcheetah.yaml
```

PPO + Ant：

```powershell
python train.py --config configs/ppo_ant.yaml
```

PPO + Humanoid：

```powershell
python train.py --config configs/ppo_humanoid.yaml
```

你也可以临时覆盖配置：

```powershell
python train.py --config configs/ppo_halfcheetah.yaml --env-id Ant-v5 --total-timesteps 100000 --seed 7
```

## 评估模型

训练结束后，把路径换成你自己的 run 目录：

```powershell
python evaluate.py --model runs\HalfCheetah-v5_PPO_seed42_YYYYMMDD_HHMMSS\final_model.zip --episodes 5
```

如果你想打开 MuJoCo 窗口渲染：

```powershell
python evaluate.py --model runs\HalfCheetah-v5_PPO_seed42_YYYYMMDD_HHMMSS\final_model.zip --episodes 3 --render
```

## 看 TensorBoard

```powershell
tensorboard --logdir runs
```

然后打开命令行显示的本地网页地址。

Windows PowerShell 脚本：

```powershell
.\scripts\open_tensorboard.ps1
```

## 作为 GitHub 示例项目

建议上传代码、配置、说明文档和实验总结，不要上传 `.venv/`、`runs/`、`.tmp/`、`__pycache__/` 这些本地或训练产物。当前 `.gitignore` 已经排除了这些目录。

更多算力预估和实验路线见 `RESULTS.md`。

正式训练前的启动清单见 `TRAINING_PLAN.md`，上传前清单见 `GITHUB_PREP.md`。

## Reward shaping 怎么做

环境本身会返回原始 reward。`rl_utils.py` 里加了 `RewardShapingWrapper`，可以在 YAML 中用 `reward_shaping.extra_info_weights` 对 info 里的 reward 分量额外加权。

示例：`configs/ppo_halfcheetah_shaped.yaml`

```yaml
reward_shaping:
  reward_scale: 1.0
  extra_info_weights:
    reward_forward: 0.10
    reward_ctrl: 0.05
```

这表示：在环境原始 reward 之外，额外鼓励向前速度，并更强地惩罚控制代价。做报告时，你可以把原始配置和 shaped 配置的曲线放在一起比较。

## Hyperparameter tuning 怎么做

优先调这些参数：

- `learning_rate`：太大容易震荡，太小学得慢。
- PPO 的 `n_steps`、`batch_size`、`n_epochs`：影响每轮采样和更新强度。
- `gamma`：越接近 1 越重视长期回报。
- `ent_coef`：鼓励探索，太大可能一直乱动。
- SAC 的 `learning_starts`、`buffer_size`、`tau`、`ent_coef`：影响 off-policy 训练稳定性。

建议实验顺序：

1. 先固定环境和算法，只改随机种子跑通。
2. 再对比 PPO 和 SAC。
3. 再改 reward shaping。
4. 最后调学习率、batch size、PPO 的 `n_steps`。

## 我建议你选择什么

先选：

- 环境：`HalfCheetah-v5`
- 算法：`PPO`
- 训练步数：先 `200000`，如果机器慢可以先 `50000`

等确认可以跑，再换 `Ant-v5`。`Humanoid-v5` 最重，不建议作为第一次跑通目标。
