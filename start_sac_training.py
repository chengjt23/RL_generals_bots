import os
import sys
from pathlib import Path
import yaml

def check_bc_model():
    experiments_dir = Path("experiments")
    if not experiments_dir.exists():
        print("未找到experiments目录")
        return None
    
    bc_dirs = sorted([d for d in experiments_dir.iterdir() if d.is_dir() and 'behavior_cloning' in d.name])
    
    if not bc_dirs:
        print("未找到行为克隆模型")
        return None
    
    print("找到以下BC训练实验：")
    for i, d in enumerate(bc_dirs, 1):
        print(f"{i}. {d.name}")
    
    latest_dir = bc_dirs[-1]
    checkpoint_dir = latest_dir / "checkpoints"
    
    if not checkpoint_dir.exists():
        print(f"在{latest_dir.name}中未找到checkpoints目录")
        return None
    
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        print(f"在{checkpoint_dir}中未找到检查点文件")
        return None
    
    best_model = checkpoint_dir / "best_model.pt"
    if best_model.exists():
        return str(best_model)
    
    latest_ckpt = sorted(checkpoints)[-1]
    return str(latest_ckpt)

def update_config(bc_model_path):
    config_path = Path("configs/config_sac.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['experiment']['bc_pretrain_path'] = bc_model_path
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"已更新配置文件，BC模型路径：{bc_model_path}")

def main():
    print("=" * 60)
    print("SAC训练启动脚本")
    print("=" * 60)
    
    bc_model_path = check_bc_model()
    
    if bc_model_path is None:
        print("\n错误：未找到BC预训练模型")
        print("请先运行行为克隆训练：python train_bc.py --config configs/config_base.yaml")
        sys.exit(1)
    
    print(f"\n使用BC模型：{bc_model_path}")
    
    update_config(bc_model_path)
    
    print("\n开始SAC训练...")
    print("-" * 60)
    
    os.system("python train_sac.py --config configs/config_sac.yaml")

if __name__ == '__main__':
    main()

