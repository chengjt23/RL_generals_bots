# Generals.io RL Training Pipeline

## 目录结构

```
RL_generals_bots/
├── agents/               # 模型定义
│   ├── __init__.py
│   ├── memory.py        # 记忆增强
│   ├── network.py       # U-Net + Policy/Value 头
│   ├── reward_shaping.py # 势函数奖励整形
│   └── sota_agent.py    # SOTA 智能体
├── configs/             # 训练配置
│   └── behavior_cloning.yaml
├── data/                # 数据加载
│   ├── dataloader.py    # 数据集和数据加载器
│   ├── download.py      # 下载脚本
│   └── generals_io_replays/  # 数据集目录
├── experiments/         # 实验输出（checkpoints, logs）
└── train_bc.py         # 行为克隆训练脚本
```

## 环境准备

### 安装依赖
```bash
pip install -r requirements.txt
```

### 下载数据集
```bash
cd RL_generals_bots
cd data
python download.py
```

## 训练流程

### 第一步：行为克隆（Behavior Cloning）

按照论文描述，使用高质量对局进行行为克隆预训练。

#### 配置说明 (`configs/config_base.yaml`)

```yaml
training:
  batch_size: 64          # 批量大小
  learning_rate: 0.0003   # 学习率
  num_epochs: 100         # 训练轮数
  gradient_clip: 1.0      # 梯度裁剪
  warmup_steps: 1000      # 预热步数
  save_top_k: 5          # 保存最好的 5 个模型

data:
  min_stars: 70          # 最低星级要求（论文设定）
  max_turns: 500         # 最大回合数（论文设定）
  grid_size: 24          # 网格大小

model:
  obs_channels: 15       # 观测通道数
  memory_channels: 0     # BC 阶段不使用记忆
  base_channels: 64      # 基础通道数
```

#### 启动训练
```bash
cd RL_generals_bots
python train_bc.py --config configs/config_base.yaml
```

#### 训练输出
训练过程会在 `experiments/` 下创建实验目录：
```
experiments/
└── behavior_cloning_20250128_143022/
    ├── config.yaml           # 保存的配置
    ├── metrics.json          # 训练指标
    └── checkpoints/          # 模型检查点
        ├── epoch_10_loss_0.1234.pt
        ├── epoch_20_loss_0.0987.pt
        └── ...（最多保存 top 5）
```

#### 监控训练
- 使用 tqdm 进度条实时显示训练进度
- `metrics.json` 记录每个 epoch 的损失和学习率
- 每个 epoch 后自动验证并保存最佳模型

## 数据集处理

### 数据过滤标准（严格按论文）
1. ✅ 游戏回合数 ≤ 500
2. ✅ 至少一名玩家星级 ≥ 70
3. ✅ 最新游戏版本

### 数据提取
- 从 replay 重建完整游戏状态
- 为每个动作提取 (observation, action, player_index) 三元组
- 自动处理动作转换（tile index → row/col/direction）

### 动作编码
- Pass: [1, 0, 0, 0, 0]
- Move: [0, row, col, direction, split]
  - direction: 0=上, 1=下, 2=左, 3=右
  - split: 0=全部, 1=一半

## 论文复现对照

| 论文描述 | 实现状态 |
|---------|---------|
| 347,000 → 16,320 高质量对局 | ✅ 过滤逻辑已实现 |
| 星级 ≥ 70 | ✅ `min_stars=70` |
| 回合 ≤ 500 | ✅ `max_turns=500` |
| 3 小时 H100 训练 | ✅ 可配置 `num_epochs` |
| 行为克隆预训练 | ✅ `train_bc.py` |
| U-Net 架构 | ✅ `network.py` |

## 使用训练好的模型

```python
from agents import SOTAAgent

# 加载检查点
agent = SOTAAgent(
    id="SOTA",
    grid_size=24,
    model_path="experiments/behavior_cloning_xxx/checkpoints/best_model.pt"
)

# 在环境中使用
from generals.envs import PettingZooGenerals
env = PettingZooGenerals(agents=["SOTA", "Random"])
obs, _ = env.reset()
action = agent.act(obs["SOTA"])
```

## 下一步：自博弈训练（待实现）

行为克隆完成后，需要实现：
1. PPO 自博弈训练循环
2. 对手池管理（保留最近 N=3 个版本）
3. GAE (λ=0.95) 优势估计
4. 势函数奖励整形

## 故障排除

### 内存不足
- 减小 `batch_size`
- 减小 `base_channels`
- 设置 `max_replays` 限制数据集大小

### 训练速度慢
- 增加 `num_workers`
- 启用 `mixed_precision: true`
- 使用更强的 GPU

### 数据加载错误
- 确保下载完整数据集
- 检查 `data_dir` 路径正确
- 尝试更新 pyarrow: `pip install -U pyarrow`

