# 学习笔记

这里记录我做这个项目时整理出来的基础概念。

## PPO 和 SAC

PPO 是 on-policy 算法。它用当前策略采样数据，再用这些数据更新当前策略。PPO 的核心是限制策略每次更新的幅度，让训练更稳定。

我对 PPO 的理解：

- 稳定性比较好。
- 适合作为 baseline。
- 样本效率一般，需要比较多环境交互。

SAC 是 off-policy 算法。它会把经验存进 replay buffer，后面可以重复使用这些数据。SAC 还会通过 entropy 鼓励探索。

我对 SAC 的理解：

- 样本效率通常更高。
- 对 replay buffer、learning starts、entropy 系数更敏感。
- 适合拿来和 PPO 做对比。

## MuJoCo 环境

我这次先选 `HalfCheetah-v5`，主要原因是它比 Ant 和 Humanoid 轻，适合先跑通。

三个环境的大致区别：

- `HalfCheetah-v5`：二维奔跑，动作维度较低，适合入门。
- `Ant-v5`：四足机器人，动作更复杂，训练更慢。
- `Humanoid-v5`：人形机器人，最难，也最吃算力。

我后续会先把 HalfCheetah 结果做稳定，再考虑 Ant。

## Reward shaping

Reward shaping 是在原始 reward 上加额外信号，让 agent 更容易学到我想要的行为。

比如 HalfCheetah 里可以更强调向前移动，也可以增加控制代价的惩罚。

这里要注意一个问题：reward shaping 可能会改变任务本身。如果设计不好，agent 可能学到奇怪的策略。所以我认为 shaping 实验必须和原始 reward baseline 对比，不能只看 shaped reward 的结果。

## Hyperparameter tuning

我目前认为调参应该按控制变量来做：

1. 固定环境和算法，先跑 baseline。
2. 固定总步数，换不同 seed。
3. 对比 PPO 和 SAC。
4. 再改 learning rate、batch size、n_steps。
5. 每次记录 TensorBoard 曲线和最终评估结果。

优先关注的参数：

- `learning_rate`
- `n_steps`
- `batch_size`
- `n_epochs`
- `gamma`
- `ent_coef`

## 这次项目给我的主要收获

我以前容易只关注算法公式。这次做完以后，我更明显感觉到 RL 项目里工程流程很重要：

- 配置要可复现。
- 训练日志要保存。
- 模型要能重新加载。
- 评估要和训练环境一致。
- 结果要解释局限，不能只报一个分数。

这也是我后面继续改进这个项目的方向。
