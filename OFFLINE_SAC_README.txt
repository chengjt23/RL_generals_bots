Offline SAC实现说明

1. 概述

这是一个基于离线replay数据的SAC算法实现，区别于在线self-play的SAC。

关键特点：
- 使用高质量的generals.io replay数据
- Off-policy + Offline训练模式
- 复用BC预训练的backbone
- 不需要与环境交互

2. 文件结构

新增文件：
- data/offline_sac_dataloader.py     数据加载器（返回SAC需要的格式）
- train_offline_sac.py               离线训练脚本
- configs/config_offline_sac.yaml    配置文件

3. Online SAC vs Offline SAC 对比

                    Online SAC (train_sac.py)    Offline SAC (train_offline_sac.py)
数据来源            实时环境交互                  历史replay数据
训练模式            Online                        Offline
对手                Self-play (对手池)            Replay中的真人玩家
数据质量            自己探索 + BC初始化           高质量真人对局
样本效率            中等（需要环境交互）          极高（数据可重复使用）
训练时间            长（环境慢）                  短（纯GPU训练）
泛化能力            取决于self-play质量           取决于replay数据多样性
适用场景            持续进化                      快速训练强基线

4. 使用方法

步骤1：确保有BC预训练模型
python train_bc.py --config configs/config_base.yaml

步骤2：确保有replay数据
数据应该在：data/generals_io_replays/

步骤3：修改配置文件
编辑 configs/config_offline_sac.yaml：
  bc_pretrain_path: "你的BC模型路径"

步骤4：开始训练
python train_offline_sac.py --config configs/config_offline_sac.yaml

5. 数据格式

Offline SAC dataloader返回：
(obs, memory, action, reward, next_obs, next_memory, done)

- obs: (15, 24, 24) 当前观察
- memory: (18, 24, 24) 历史信息
- action: (5,) [is_pass, row, col, direction, split]
- reward: float 奖励（使用PotentialBasedRewardFn计算）
- next_obs: (15, 24, 24) 下一个观察
- next_memory: (18, 24, 24) 下一个历史信息
- done: float 是否结束

6. 关键区别说明

数据收集：
  Online: 每步调用 env.step(action) 获取新数据
  Offline: 从parquet文件读取历史数据

训练循环：
  Online: while training: collect → store → sample → update
  Offline: for epoch: for batch in dataloader: update

对手：
  Online: RandomAgent或Self-play对手池
  Offline: Replay中的真人玩家（通常更强）

7. 优势与局限

优势：
✓ 利用高质量专家数据
✓ 训练速度快（无环境交互）
✓ BC backbone + SAC fine-tune
✓ 数据可重复使用
✓ 适合资源有限的场景

局限：
✗ 受限于replay数据质量
✗ 无法探索新策略
✗ 可能出现分布偏移
✗ 需要大量存储空间

8. 预期效果

训练时间：
  10 epochs × 数据量 ≈ 2-5小时（取决于replay数量）

性能预期：
  vs Random: 85-90%
  vs BC: 50-60%
  泛化能力: 中等（取决于replay多样性）

对比Online SAC：
  训练时间: 快5-10倍
  最终性能: 可能略低（无self-play进化）
  稳定性: 更稳定（数据固定）

9. 参数调整建议

学习率（比online小）：
  actor_lr: 0.0001  （online是0.0003）
  critic_lr: 0.0001 （online是0.0003）
  原因：离线数据分布固定，不需要快速适应

Epochs：
  start: 5-10 epochs
  如果loss还在下降，可以继续训练

Batch size：
  推荐: 128-256
  GPU允许的话可以更大

10. 使用场景建议

选择Offline SAC如果：
✓ 有高质量replay数据
✓ 希望快速训练baseline
✓ 资源有限（单机）
✓ 需要稳定的训练过程

选择Online SAC如果：
✓ 追求最高性能
✓ 需要持续进化
✓ 有足够训练时间
✓ 可以承受环境交互成本

推荐策略：
1. 先用Offline SAC快速训练baseline（2-3小时）
2. 再用Online SAC继续提升（作为warm start）
3. 结合两者优势

11. 运行命令总结

训练：
python train_offline_sac.py --config configs/config_offline_sac.yaml

评估：
python evaluate_agents.py 
  --bc_model path/to/bc.pt 
  --sac_checkpoint experiments/offline_sac_xxx/checkpoints/final.pt

运行游戏：
cd running
python run_sac_agent.py 
  --checkpoint ../experiments/offline_sac_xxx/checkpoints/final.pt

12. 技术细节

Replay数据处理：
- 每个replay包含完整对局
- 按turn顺序处理每个move
- 计算reward使用PotentialBasedRewardFn
- Memory自动更新和padding

BC Backbone复用：
- Actor加载：backbone + policy_head
- Critic加载：backbone（初始化）
- 继承BC的特征提取能力

SAC算法：
- 使用相同的update逻辑
- 期望形式（离散动作）
- 自动α调整
- 双Q网络

13. 故障排除

问题：数据加载慢
解决：增加num_workers（但注意内存）

问题：GPU内存不足
解决：减小batch_size

问题：Loss不下降
解决：检查学习率，可能需要降低

问题：性能不如BC
解决：正常，前几个epoch可能略差，继续训练

14. 下一步改进

可选优化：
1. 实现CQL（Conservative Q-Learning）- 防止Q值过估计
2. 实现IQL（Implicit Q-Learning）- 更稳定的offline RL
3. 加入数据增强
4. 在线fine-tuning（offline → online过渡）

15. 总结

这是一个完整的Offline SAC实现，利用高质量replay数据进行训练。
与Online SAC互补，提供快速训练高质量baseline的能力。

两个版本可以独立使用，也可以组合使用：
  Offline SAC（快速baseline）→ Online SAC（持续优化）

立即开始训练！

