import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json

from data.dataloader import GeneralsReplayDataset
from data.iterable_dataloader import create_iterable_dataloader
from agents.network import SOTANetwork

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Logging to wandb is disabled.")


class BehaviorCloningTrainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(self.config['training']['device'])
        self.setup_seed()
        self.setup_dirs()
        self.setup_model()
        self.setup_wandb()
        self.setup_data()
        self.setup_optimizer()
        self.setup_training()
    
    def setup_seed(self):
        seed = self.config['seed']
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
    def setup_dirs(self):
        exp_name = self.config['logging']['experiment_name']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = Path(self.config['logging']['save_dir']) / f"{exp_name}_{timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        self.ckpt_dir = self.exp_dir / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)
        
        with open(self.exp_dir / "config.yaml", 'w') as f:
            yaml.dump(self.config, f)
        
        print(f"Experiment directory: {self.exp_dir}")
    
    def setup_model(self):
        model_config = self.config['model']
        self.model = SOTANetwork(
            obs_channels=model_config['obs_channels'],
            memory_channels=model_config['memory_channels'],
            grid_size=model_config['grid_size'],
            base_channels=model_config['base_channels'],
            rnn_hidden_channels=model_config.get('rnn_hidden_channels', 32),
            rnn_num_layers=model_config.get('rnn_num_layers', 2),
            use_rnn_memory=model_config.get('use_rnn_memory', True),
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def setup_wandb(self):
        if WANDB_AVAILABLE and self.config['logging'].get('use_wandb', False):
            exp_name = self.config['logging']['experiment_name']
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            wandb.init(
                project=self.config['logging'].get('wandb_project', 'generals-rl'),
                name=f"{exp_name}_{timestamp}",
                config=self.config,
                dir=str(self.exp_dir),
                tags=self.config['logging'].get('wandb_tags', []),
            )
            wandb.watch(self.model, log='all', log_freq=100)
            print(f"Wandb initialized: {wandb.run.url}")
        else:
            if not WANDB_AVAILABLE:
                print("Wandb not available (not installed)")
            else:
                print("Wandb disabled in config")
    
    def setup_data(self):
        # print data configs
        print("Data configuration:")
        print(yaml.dump(self.config['data']))

        data_config = self.config['data']
        train_config = self.config['training']
        
        train_replays = int(data_config['max_replays'] * data_config['train_split'])
        val_replays = data_config['max_replays'] - train_replays

        # print train and val replays
        print(f"Train replays (streaming): {train_replays}")
        print(f"Validation replays: {val_replays}")
        
        self.train_loader = create_iterable_dataloader(
            data_dir=data_config['data_dir'],
            batch_size=train_config['batch_size'],
            grid_size=data_config['grid_size'],
            num_workers=train_config['num_workers'],
            max_replays=train_replays,
            min_stars=data_config['min_stars'],
            max_turns=data_config['max_turns'],
            sequence_length=train_config.get('sequence_length', 32),
        )
        
        self.val_loader = create_iterable_dataloader(
            data_dir=data_config['data_dir'],
            batch_size=train_config['batch_size'],
            grid_size=data_config['grid_size'],
            num_workers=train_config['num_workers'],
            max_replays=val_replays,
            min_stars=data_config['min_stars'],
            max_turns=data_config['max_turns'],
            sequence_length=train_config.get('sequence_length', 32),
            skip_replays=train_replays,
        )
        
        print(f"Train replays (streaming): {train_replays}")
        print(f"Val replays (streaming): {val_replays}")
    
    def setup_optimizer(self):
        opt_config = self.config['optimizer']
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            betas=opt_config['betas'],
            eps=opt_config['eps'],
            weight_decay=self.config['training']['weight_decay'],
        )
        
        scheduler_config = self.config['scheduler']
        
        self.steps_per_epoch = self.config['training'].get('steps_per_epoch', 20000)
        
        num_training_steps = self.steps_per_epoch * self.config['training']['num_epochs']
        warmup_steps = self.config['training']['warmup_steps']
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            min_lr = scheduler_config['min_lr']
            base_lr = self.config['training']['learning_rate']
            return max(min_lr / base_lr, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def setup_training(self):
        self.global_step = 0
        self.best_losses = []
        self.scaler = GradScaler() if self.config['training']['mixed_precision'] else None
        
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        }
        
        early_stop_config = self.config['training'].get('early_stopping', {})
        self.early_stopping_enabled = early_stop_config.get('enabled', True)
        self.early_stopping_patience = early_stop_config.get('patience', 10)
        self.early_stopping_min_delta = early_stop_config.get('min_delta', 0.0)
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
    
    def compute_loss(self, obs, actions, policy_logits):
        batch_size = obs.shape[0]
        grid_size = self.config['model']['grid_size']
        
        pass_flag = actions[:, 0]
        row = actions[:, 1]
        col = actions[:, 2]
        direction = actions[:, 3]
        split = actions[:, 4]
        
        policy_logits_reshaped = policy_logits.view(batch_size, 9, grid_size, grid_size)
        
        pass_logits = policy_logits_reshaped[:, 0, 0, 0]
        action_logits = policy_logits_reshaped[:, 1:, :, :].permute(0, 2, 3, 1).reshape(batch_size, -1, 8)
        
        losses = []
        for i in range(batch_size):
            if pass_flag[i] == 1:
                target = torch.zeros(1 + grid_size * grid_size * 8, device=self.device)
                target[0] = 1.0
            else:
                r, c, d, s = int(row[i]), int(col[i]), int(direction[i]), int(split[i])
                if r >= grid_size or c >= grid_size:
                    continue
                
                flat_idx = r * grid_size + c
                action_idx = d * 2 + s
                
                target = torch.zeros(1 + grid_size * grid_size * 8, device=self.device)
                target[1 + flat_idx * 8 + action_idx] = 1.0
            
            all_logits = torch.cat([
                pass_logits[i:i+1],
                action_logits[i].flatten()
            ])
            
            log_probs = F.log_softmax(all_logits, dim=0)
            loss = F.kl_div(log_probs, target, reduction='sum')
            losses.append(loss)
        
        if len(losses) == 0:
            return torch.zeros(batch_size, device=self.device)
        
        return torch.stack(losses)
    
    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.config['training']['num_epochs']} [Train]",
            unit="batch",
            leave=False,
            total=self.steps_per_epoch
        )
        
        for batch_idx, (obs_seq, actions_seq, lengths) in enumerate(pbar):
            # obs_seq: (B, L, C, H, W)
            # actions_seq: (B, L, 5)
            # lengths: (B,)
            obs_seq = obs_seq.to(self.device)
            actions_seq = actions_seq.to(self.device)
            lengths = lengths.to(self.device)
            
            batch_size, seq_len = obs_seq.shape[:2]
            
            self.optimizer.zero_grad()
            
            # Initialize hidden state for the batch
            hidden_state = None
            
            # Truncated Backpropagation Through Time (TBPTT)
            tbptt_steps = self.config['training'].get('tbptt_steps', 32)
            
            for t_start in range(0, seq_len, tbptt_steps):
                t_end = min(t_start + tbptt_steps, seq_len)
                
                # Check if any sequence in batch is still valid
                if (lengths > t_start).sum() == 0:
                    break
                
                chunk_loss = 0.0
                valid_steps_in_chunk = 0
                
                # Process chunk
                for t in range(t_start, t_end):
                    mask = (t < lengths).float()
                    if mask.sum() == 0:
                        break
                    
                    obs_t = obs_seq[:, t]
                    actions_t = actions_seq[:, t]
                    
                    if self.scaler is not None:
                        with autocast():
                            policy_logits, _, hidden_state = self.model(
                                obs_t, 
                                hidden_state=hidden_state, 
                                return_hidden=True
                            )
                            loss_t = self.compute_loss(obs_t, actions_t, policy_logits)
                            loss_t = (loss_t * mask).sum()
                            chunk_loss += loss_t
                    else:
                        policy_logits, _, hidden_state = self.model(
                            obs_t, 
                            hidden_state=hidden_state, 
                            return_hidden=True
                        )
                        loss_t = self.compute_loss(obs_t, actions_t, policy_logits)
                        loss_t = (loss_t * mask).sum()
                        chunk_loss += loss_t
                    
                    valid_steps_in_chunk += mask.sum()
                
                # Normalize loss for the chunk
                if valid_steps_in_chunk > 0:
                    chunk_loss = chunk_loss / valid_steps_in_chunk
                    
                    # Backward pass for the chunk
                    if self.scaler is not None:
                        self.scaler.scale(chunk_loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['training']['gradient_clip']
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        chunk_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['training']['gradient_clip']
                        )
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    
                    # Detach hidden state for next chunk
                    if hidden_state is not None:
                        hidden_state = [(h.detach(), c.detach()) for h, c in hidden_state]
                    
                    total_loss += chunk_loss.item()
            
            self.scheduler.step()
            num_batches += 1
            
            if num_batches >= self.steps_per_epoch:
                break
            
            avg_loss = total_loss / num_batches
            pbar.set_postfix({
                'loss': f"{chunk_loss.item():.4f}",
                'avg_loss': f"{avg_loss:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            if WANDB_AVAILABLE and self.config['logging'].get('use_wandb', False):
                wandb.log({
                    'train/loss': chunk_loss.item(),
                    'train/avg_loss': avg_loss,
                    'train/learning_rate': self.scheduler.get_last_lr()[0],
                    'train/step': self.global_step,
                }, step=self.global_step)
            
            self.global_step += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            self.val_loader,
            desc="Validation",
            unit="batch",
            ncols=120,
            leave=False
        )
        
        for obs_seq, actions_seq, lengths in pbar:
            obs_seq = obs_seq.to(self.device)
            actions_seq = actions_seq.to(self.device)
            lengths = lengths.to(self.device)
            
            batch_size, seq_len = obs_seq.shape[:2]
            
            # Initialize hidden state for the batch
            hidden_state = None
            
            # Truncated Backpropagation Through Time (TBPTT)
            tbptt_steps = self.config['training'].get('tbptt_steps', 32)
            
            for t_start in range(0, seq_len, tbptt_steps):
                t_end = min(t_start + tbptt_steps, seq_len)
                
                # Check if any sequence in batch is still valid
                if (lengths > t_start).sum() == 0:
                    break
                
                chunk_loss = 0.0
                valid_steps_in_chunk = 0
                
                # Process chunk
                for t in range(t_start, t_end):
                    mask = (t < lengths).float()
                    if mask.sum() == 0:
                        break
                    
                    obs_t = obs_seq[:, t]
                    actions_t = actions_seq[:, t]
                    
                    policy_logits, _, hidden_state = self.model(
                        obs_t, 
                        hidden_state=hidden_state, 
                        return_hidden=True
                    )
                    loss_t = self.compute_loss(obs_t, actions_t, policy_logits)
                    loss_t = (loss_t * mask).sum()
                    chunk_loss += loss_t
                    
                    valid_steps_in_chunk += mask.sum()
                
                # Normalize loss for the chunk
                if valid_steps_in_chunk > 0:
                    chunk_loss = chunk_loss / valid_steps_in_chunk
                    
                    # Detach hidden state for next chunk
                    if hidden_state is not None:
                        hidden_state = [(h.detach(), c.detach()) for h, c in hidden_state]
                    
                    total_loss += chunk_loss.item()
            
            num_batches += 1
            
            avg_loss = total_loss / num_batches
            pbar.set_postfix({'val_loss': f"{avg_loss:.4f}"})
        
        final_avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        if WANDB_AVAILABLE and self.config['logging'].get('use_wandb', False):
            wandb.log({
                'val/loss': final_avg_loss,
                'val/step': self.global_step,
            }, step=self.global_step)
        
        return final_avg_loss
    
    def save_checkpoint(self, epoch: int, val_loss: float):
        ckpt_path = self.ckpt_dir / f"epoch_{epoch}_loss_{val_loss:.4f}.pt"
        
        torch.save({
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
        }, ckpt_path)
        
        self.best_losses.append((val_loss, ckpt_path))
        self.best_losses.sort(key=lambda x: x[0])
        
        if len(self.best_losses) > self.config['training']['save_top_k']:
            _, old_ckpt = self.best_losses.pop()
            if old_ckpt.exists():
                old_ckpt.unlink()
        
        print(f"Saved checkpoint: {ckpt_path}")
    
    def train(self):
        print("\n" + "="*60)
        print(" " * 15 + "Behavior Cloning Training")
        print("="*60)
        print(f"Experiment: {self.exp_dir.name}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config['training']['num_epochs']}")
        print(f"Steps per epoch: {self.steps_per_epoch}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        print(f"Learning rate: {self.config['training']['learning_rate']}")
        if self.early_stopping_enabled:
            print(f"Early stopping: enabled (patience={self.early_stopping_patience}, min_delta={self.early_stopping_min_delta})")
        else:
            print(f"Early stopping: disabled")
        print("="*60 + "\n")
        
        num_epochs = self.config['training']['num_epochs']
        
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            self.metrics['train_loss'].append(train_loss)
            self.metrics['val_loss'].append(val_loss)
            self.metrics['learning_rate'].append(self.scheduler.get_last_lr()[0])
            
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"LR: {self.scheduler.get_last_lr()[0]:.2e}")
            
            if WANDB_AVAILABLE and self.config['logging'].get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'epoch/train_loss': train_loss,
                    'epoch/val_loss': val_loss,
                    'epoch/learning_rate': self.scheduler.get_last_lr()[0],
                }, step=self.global_step)
            
            self.save_checkpoint(epoch, val_loss)
            
            with open(self.exp_dir / "metrics.json", 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            if self.early_stopping_enabled:
                if val_loss < self.best_val_loss - self.early_stopping_min_delta:
                    self.best_val_loss = val_loss
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
                    if self.early_stopping_counter >= self.early_stopping_patience:
                        print(f"\nEarly stopping triggered after {epoch} epochs")
                        print(f"Best validation loss: {self.best_val_loss:.4f}")
                        break
        
        if WANDB_AVAILABLE and self.config['logging'].get('use_wandb', False):
            wandb.log({
                'best_val_loss': min(self.metrics['val_loss']),
            })
            wandb.finish()
        
        print("\n" + "="*60)
        print(" " * 20 + "Training Complete!")
        print("="*60)
        print(f"✓ Best checkpoints saved in: {self.ckpt_dir}")
        print(f"✓ Best validation loss: {min(self.metrics['val_loss']):.4f}")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='/root/shared-nvme/oyx/bc_v2/configs/config_base.yaml',
        help='Path to config file'
    )
    args = parser.parse_args()
    
    trainer = BehaviorCloningTrainer(args.config)
    trainer.train()


if __name__ == '__main__':
    main()

