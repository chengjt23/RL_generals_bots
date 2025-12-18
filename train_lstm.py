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
    import swanlab as wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Logging to wandb is disabled.")


class LSTMOnlyTrainer:
    def __init__(self, config_path: str, checkpoint_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.checkpoint_path = checkpoint_path
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
        exp_name = self.config['logging']['experiment_name'] + "_lstm_only"
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
        ).to(self.device)
        
        # Load checkpoint
        print(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # Handle state dict mismatch
        # The checkpoint might be from the original UNet (no LSTM), so 'backbone.conv_lstm' keys will be missing.
        state_dict = checkpoint['model_state_dict']
        
        # Check if we need to adapt keys (e.g. if the checkpoint has 'module.' prefix from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict

        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        
        # Verify that missing keys are only related to the new LSTM module
        lstm_missing = [k for k in missing_keys if 'conv_lstm' in k]
        other_missing = [k for k in missing_keys if 'conv_lstm' not in k]
        
        if len(lstm_missing) > 0:
            print(f"Initializing new LSTM weights (missing in checkpoint): {len(lstm_missing)} keys")
            # Verify that we are indeed missing the LSTM weights we expect
            # ConvLSTM has weight_ih, weight_hh, bias_ih, bias_hh
            # It might be wrapped in a cell or not depending on implementation
            print(f"Missing LSTM keys: {lstm_missing}")
        
        if len(other_missing) > 0:
            print(f"WARNING: Found unexpected missing keys: {other_missing}")
        
        if len(unexpected_keys) > 0:
            print(f"WARNING: Found unexpected keys in checkpoint: {unexpected_keys}")

        print(f"Successfully loaded backbone weights.")
        
        # Freeze everything except LSTM
        for name, param in self.model.named_parameters():
            if 'conv_lstm' in name:
                param.requires_grad = True
                print(f"Unfrozen: {name}")
            else:
                param.requires_grad = False
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def setup_wandb(self):
        if WANDB_AVAILABLE and self.config['logging'].get('use_wandb', False):
            exp_name = self.config['logging']['experiment_name'] + "_lstm_only"
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            wandb.init(
                project=self.config['logging'].get('wandb_project', 'generals-rl'),
                workspace=self.config['logging'].get('wandb_entity', None),
                name=f"{exp_name}_{timestamp}",
                config=self.config,
                dir=str(self.exp_dir),
                tags=self.config['logging'].get('wandb_tags', []) + ['lstm_only'],
            )
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
        )
        
        # Create persistent iterator for resuming
        self.train_iterator = iter(self.train_loader)
        
        self.val_loader = create_iterable_dataloader(
            data_dir=data_config['data_dir'],
            batch_size=1, # Validate one game at a time
            grid_size=data_config['grid_size'],
            num_workers=1,
            max_replays=val_replays,
            min_stars=data_config['min_stars'],
            max_turns=data_config['max_turns'],
            sequence_len=32,
        )
        
        print(f"Train replays (streaming): {train_replays}")
        print(f"Val replays (streaming): {val_replays}")
    
    def setup_optimizer(self):
        opt_config = self.config['optimizer']
        # Only optimize trainable parameters (LSTM)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
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
        # obs shape: (B, T, C, H, W)
        # actions shape: (B, T, 5)
        # policy_logits shape: (B, T, 9, H, W)
        
        batch_size = obs.shape[0]
        seq_len = obs.shape[1]
        grid_size = self.config['model']['grid_size']
        
        # Flatten batch and time dimensions for loss computation
        # actions: (B*T, 5)
        actions = actions.view(-1, 5)
        
        pass_flag = actions[:, 0]
        row = actions[:, 1]
        col = actions[:, 2]
        direction = actions[:, 3]
        split = actions[:, 4]
        
        # policy_logits: (B*T, 9, H, W)
        policy_logits_reshaped = policy_logits.view(batch_size * seq_len, 9, grid_size, grid_size)
        
        pass_logits = policy_logits_reshaped[:, 0, 0, 0]
        action_logits = policy_logits_reshaped[:, 1:, :, :].permute(0, 2, 3, 1).reshape(batch_size * seq_len, -1, 8)
        
        losses = []
        total_samples = batch_size * seq_len
        
        for i in range(total_samples):
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
            return torch.tensor(0.0, device=self.device)
        
        return torch.stack(losses).mean()
    
    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Dictionary to store hidden states for each worker
        # Key: worker_id (int), Value: (h, c) tuple
        if not hasattr(self, 'hidden_states'):
            self.hidden_states = {}
        
        pbar = tqdm(
            range(self.steps_per_epoch),
            desc=f"Epoch {epoch}/{self.config['training']['num_epochs']} [Train]",
            unit="batch",
            leave=False,
        )
        
        for i, _ in enumerate(pbar):
            try:
                # print(f"DEBUG: Waiting for batch {i}")
                obs, memory, actions, reset_mask, worker_id = next(self.train_iterator)
                if i % 100 == 0:
                    print(f"DEBUG: Got batch {i} from worker {worker_id.item()}")
            except StopIteration:
                self.train_iterator = iter(self.train_loader)
                obs, memory, actions, reset_mask, worker_id = next(self.train_iterator)
            
            # worker_id is a tensor of shape (1,) or scalar because DataLoader collates
            # Since batch_size=None in DataLoader, it yields exactly what dataset yields.
            # Dataset yields (..., worker_id_tensor) where worker_id_tensor is scalar tensor.
            curr_w_id = worker_id.item()
                
            obs = obs.to(self.device)
            memory = memory.to(self.device)
            actions = actions.to(self.device)
            reset_mask = reset_mask.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Retrieve hidden state for this worker
            if curr_w_id not in self.hidden_states:
                self.hidden_states[curr_w_id] = None
            
            current_hidden_state = self.hidden_states[curr_w_id]
            
            if current_hidden_state is not None:
                h, c = current_hidden_state
                # Ensure hidden state is on correct device (in case it was initialized on CPU or moved)
                h = h.to(self.device)
                c = c.to(self.device)
                
                mask_expanded = reset_mask.view(-1, 1, 1, 1).expand_as(h)
                h = h * (~mask_expanded)
                c = c * (~mask_expanded)
                current_hidden_state = (h, c)
            
            if self.scaler is not None:
                with autocast():
                    policy_logits, _, new_hidden_state = self.model(obs, memory, current_hidden_state)
                    loss = self.compute_loss(obs, actions, policy_logits)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                policy_logits, _, new_hidden_state = self.model(obs, memory, current_hidden_state)
                loss = self.compute_loss(obs, actions, policy_logits)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
                self.optimizer.step()
            
            h, c = new_hidden_state
            # Detach and store for next iteration of THIS worker
            self.hidden_states[curr_w_id] = (h.detach(), c.detach())
            
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            avg_loss = total_loss / num_batches
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{avg_loss:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            if WANDB_AVAILABLE and self.config['logging'].get('use_wandb', False):
                wandb.log({
                    'train/loss': loss.item(),
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
        
        val_iterator = iter(self.val_loader)
        val_hidden_state = None
        val_steps = 1000 
        
        pbar = tqdm(
            range(val_steps),
            desc="Validation",
            unit="chunk",
            ncols=120,
            leave=False
        )
        
        for _ in pbar:
            try:
                # Validation loader also yields worker_id now, but we ignore it for single-stream validation
                # Or we can use it if we want multi-stream validation
                batch_data = next(val_iterator)
                if len(batch_data) == 5:
                    obs, memory, actions, reset_mask, _ = batch_data
                else:
                    obs, memory, actions, reset_mask = batch_data
            except StopIteration:
                break
                
            obs = obs.to(self.device)
            memory = memory.to(self.device)
            actions = actions.to(self.device)
            reset_mask = reset_mask.to(self.device)
            
            if val_hidden_state is not None:
                h, c = val_hidden_state
                mask_expanded = reset_mask.view(-1, 1, 1, 1).expand_as(h)
                h = h * (~mask_expanded)
                c = c * (~mask_expanded)
                val_hidden_state = (h, c)
            
            policy_logits, _, new_hidden_state = self.model(obs, memory, val_hidden_state)
            loss = self.compute_loss(obs, actions, policy_logits)
            
            val_hidden_state = new_hidden_state
            
            total_loss += loss.item()
            num_batches += 1
            
            avg_loss = total_loss / num_batches
            pbar.set_postfix({'val_loss': f"{avg_loss:.4f}"})
        
        if num_batches == 0:
            return 0.0
            
        final_avg_loss = total_loss / num_batches
        
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
        print(" " * 15 + "LSTM-Only Training")
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
            
            if val_loss < self.best_val_loss - self.early_stopping_min_delta:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                self.save_checkpoint(epoch, val_loss)
            else:
                self.early_stopping_counter += 1
                print(f"Early stopping counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
                if self.early_stopping_enabled and self.early_stopping_counter >= self.early_stopping_patience:
                    print("Early stopping triggered")
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM module only")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained UNet checkpoint")
    args = parser.parse_args()
    
    trainer = LSTMOnlyTrainer(args.config, args.checkpoint)
    trainer.train()