SAC算法实现 - 完整说明

项目概述

本项目实现了基于BC（行为克隆）预训练的SAC（Soft Actor-Critic）强化学习算法
用于Generals.io游戏的智能体训练

核心思路

1. 利用BC预训练模型作为初始化
2. 通过SAC算法进行在线强化学习
3. 持续优化策略超越BC性能

主要优势

从好策略开始：BC预训练提供良好初始化
高样本效率：Off-policy算法，经验回放
自动探索：最大熵框架，平衡探索利用
训练稳定：双Q网络，减少过估计

快速开始

三行命令启动：

python run_sac_test.py
python start_sac_training.py
python evaluate_agents.py --bc_model path/to/bc.pt --sac_checkpoint path/to/sac.pt

详细使用

1. 测试环境
python run_sac_test.py

2. 配置训练
编辑 configs/config_sac.yaml
设置 bc_pretrain_path 为你的BC模型路径

3. 开始训练
python train_sac.py --config configs/config_sac.yaml

4. 评估性能
python evaluate_agents.py --bc_model BC路径 --sac_checkpoint SAC路径 --num_games 20

5. 运行游戏
cd running
python run_sac_agent.py --checkpoint SAC路径 --num_games 10

文件结构

核心实现
agents/sac_network.py - Actor和Critic网络
agents/sac_agent.py - SAC智能体
agents/replay_buffer.py - 经验回放

训练评估
train_sac.py - 训练脚本
evaluate_agents.py - 评估脚本
run_sac_test.py - 测试脚本

配置文档
configs/config_sac.yaml - 配置文件
SAC_USAGE.txt - 详细说明
SAC_QUICK_START.txt - 快速指南

关键参数

gamma: 0.99 - 折扣因子
tau: 0.005 - 目标网络更新率
alpha: 0.2 - 温度参数初始值
batch_size: 128 - 批量大小
buffer_size: 100000 - 回放缓冲区大小
learning_starts: 10000 - 开始学习的步数

训练建议

1. 确保BC模型已训练好
2. 从小规模测试开始
3. 监控训练曲线
4. 定期保存检查点
5. 至少训练500k步
6. 对比BC和SAC性能

预期结果

初始阶段（0-50k步）：
- 性能可能略有下降（探索期）
- 策略熵较高
- Q值逐渐稳定

中期阶段（50k-200k步）：
- 性能开始提升
- 超越BC基线
- 策略更加稳定

后期阶段（200k+步）：
- 显著超越BC
- 策略熵降低
- 性能稳定提升

监控指标

训练指标：
- critic_loss: Critic损失
- actor_loss: Actor损失  
- alpha: 温度参数
- entropy: 策略熵

评估指标：
- episode_reward: 每局奖励
- avg_reward_10: 10局平均奖励
- episode_length: 每局步数
- win_rate: 胜率

常见问题

Q: 显存不足？
A: 减小batch_size和buffer_size

Q: 训练太慢？
A: 增加update_frequency，减少gradient_steps

Q: 性能下降？
A: 正常现象，继续训练会恢复并超越

Q: BC权重未加载？
A: 检查路径，确保文件存在

Q: 想继续训练？
A: 加载检查点，继续运行train_sac.py

技术细节

网络架构：
- 复用BC的UNet backbone
- Actor输出9通道logits
- Critic输出9通道Q值
- 双Critic + Target网络

更新机制：
- Critic: TD误差最小化
- Actor: Q值和熵联合优化
- Alpha: 自动调整温度
- Target: 软更新

动作处理：
- 支持valid action masking
- Categorical分布采样
- 确定性/随机模式

经验回放：
- 循环缓冲区
- 随机采样
- GPU加速

代码特点

简洁：约830行核心代码
高效：GPU加速，批量处理
模块化：易于扩展和修改
文档齐全：4个说明文档

性能优化

1. 使用GPU训练
2. 调整batch_size平衡速度和稳定性
3. 调整learning_starts控制探索
4. 使用wandb监控训练
5. 定期评估选择最佳模型

进阶优化

1. 优先经验回放（PER）
2. N-step returns
3. 分布式训练
4. 自对弈训练
5. Curriculum learning
6. 多任务学习

贡献

本实现特色：
- 完整的SAC实现
- BC预训练集成
- 详细的文档
- 丰富的工具脚本
- 生产级代码质量

参考资料

SAC论文：Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL
BC论文：Learning from Demonstrations
环境：Generals.io

支持

问题反馈：检查文档和常见问题
性能问题：调整超参数
实现问题：查看代码注释

总结

完整的SAC实现
基于BC预训练
即开即用
详细文档
易于扩展

立即开始你的SAC训练之旅！

