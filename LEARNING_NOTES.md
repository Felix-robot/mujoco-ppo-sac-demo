# RL 任务学习笔记

## PPO 和 SAC 的区别

PPO 是 on-policy 算法。它每次用当前策略采样一批数据，然后限制策略更新幅度，稳定但样本效率一般。适合先跑通实验，也比较容易 debug。

SAC 是 off-policy 算法。它把历史经验放进 replay buffer 反复利用，同时最大化 reward 和 entropy。它通常样本效率更高，但对 replay buffer、warmup 步数、entropy 系数更敏感。

## MuJoCo 环境怎么选

`HalfCheetah-v5`：二维奔跑，状态和动作维度相对小，最适合第一次跑通。

`Ant-v5`：四足机器人，动作维度更高，训练更慢，但展示效果更好。

`Humanoid-v5`：人形机器人，最难也最耗算力，不适合作为第一次验证环境。

## Reward shaping

Reward shaping 是在原始奖励基础上加入额外信号，让智能体更快学到你想要的行为。例如在 locomotion 任务里，可以提高向前速度奖励、加重能量消耗惩罚，或给摔倒行为更强惩罚。

要小心：reward shaping 可能让 agent 学会“钻空子”。比如只追求速度可能导致动作很抖，控制惩罚太重又可能导致 agent 不敢动。所以建议每次只改一两个 shaping 项，并保存对比曲线。

## Hyperparameter tuning

调参不是随便改参数，而是控制变量地比较实验：

1. 先固定环境、算法、总步数，只换 seed，判断结果波动。
2. 再换算法，比如 PPO 对比 SAC。
3. 再改 learning rate，例如 `3e-4`、`1e-4`、`1e-3`。
4. PPO 可以调 `n_steps` 和 `batch_size`；SAC 可以调 `learning_starts` 和 `ent_coef`。
5. 每次实验记录 mean reward 曲线，不只看最后一次结果。

## 报告可以写什么

最简单的报告结构：

1. 实验环境：HalfCheetah-v5，连续动作控制任务。
2. 算法：PPO 或 SAC，以及为什么选它。
3. 训练设置：总步数、seed、主要超参数。
4. 结果：TensorBoard reward 曲线、最终评估 reward。
5. 分析：reward shaping 前后有什么变化，哪些超参数最敏感。
