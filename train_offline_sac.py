import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from agents.sac_agent import SACAgent
from data.offline_sac_dataloader import create_offline_sac_dataloader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class OfflineSACTrainer:
    def __init__(self, config_path: str, resume_checkpoint: str = None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = self.config['training']['device']
        self.resume_checkpoint = resume_checkpoint
        self.setup_seed()
        self.setup_dirs()
        self.setup_agent()
        self.setup_data()
        self.setup_wandb()
        
        self.start_epoch = 0
        self.start_step = 0
        if self.resume_checkpoint:
            self.load_checkpoint()
    
    def setup_seed(self):
        seed = self.config['training']['seed']
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
    def setup_dirs(self):
        exp_name = self.config['experiment']['name']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = Path(self.config['experiment']['save_dir']) / f"{exp_name}_{timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        with open(self.exp_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)
    
    def setup_agent(self):
        from agents.sac_network import initialize_critic_from_bc_value
        
        # Check if BC initialization should be used
        use_bc_init = self.config['experiment'].get('use_bc_init', True)
        
        # If using BC init, pass bc_model_path; otherwise pass None
        bc_model_path = self.config['experiment'].get('bc_pretrain_path') if use_bc_init else None
        
        self.agent = SACAgent(
            id="OfflineSAC",
            grid_size=self.config['model']['grid_size'],
            device=self.device,
            bc_model_path=bc_model_path,
            memory_channels=self.config['model']['memory_channels'],
            gamma=self.config['training']['gamma'],
            tau=self.config['training']['tau'],
            alpha=self.config['training']['alpha'],
            auto_tune_alpha=self.config['training']['auto_tune_alpha'],
            actor_lr=self.config['training']['actor_lr'],
            critic_lr=self.config['training']['critic_lr'],
            alpha_lr=self.config['training']['alpha_lr'],
            # Offline RL parameters
            cql_alpha=self.config['training'].get('cql_alpha', 0.0),
            bc_weight=self.config['training'].get('bc_weight', 0.0),
            gradient_clip=self.config['training'].get('gradient_clip', 1.0),
        )
        
        # Only initialize critics from BC if BC init is enabled and CQL/BC regularization is enabled
        if use_bc_init and (self.config['training'].get('cql_alpha', 0) > 0 or 
                            self.config['training'].get('bc_weight', 0) > 0):
            bc_path = self.config['experiment']['bc_pretrain_path']
            print("\nInitializing critics from BC Value Head...")
            initialize_critic_from_bc_value(self.agent.critic_1, bc_path, self.device)
            initialize_critic_from_bc_value(self.agent.critic_2, bc_path, self.device)
            
            # Copy to target networks
            self.agent.critic_1_target.load_state_dict(self.agent.critic_1.state_dict())
            self.agent.critic_2_target.load_state_dict(self.agent.critic_2.state_dict())
            print("Initialized target networks from critics\n")
        
        if use_bc_init:
            print(f"Loaded BC pretrained weights from: {self.config['experiment']['bc_pretrain_path']}")
        else:
            print("Using random initialization for all networks")
    
    def setup_data(self):
        self.dataloader = create_offline_sac_dataloader(
            data_dir=self.config['data']['data_dir'],
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            grid_size=self.config['model']['grid_size'],
            max_replays=self.config['data'].get('max_replays'),
            min_stars=self.config['data'].get('min_stars', 70),
        )
    
    def setup_wandb(self):
        if WANDB_AVAILABLE and self.config['logging']['use_wandb']:
            wandb.init(
                project=self.config['logging']['project_name'],
                entity=self.config['logging']['wandb_entity'],
                name=self.exp_dir.name,
                config=self.config
            )
            self.use_wandb = True
        else:
            self.use_wandb = False
    
    def train(self):
        total_epochs = self.config['training']['epochs']
        log_frequency = self.config['training']['log_frequency']
        save_frequency = self.config['training']['save_frequency']
        steps_per_epoch = self.config['training'].get('steps_per_epoch', None)
        
        global_step = self.start_step
        
        use_bc_init = self.config['experiment'].get('use_bc_init', True)
        
        print("\n" + "="*60)
        print("Offline SAC Training with CQL + BC Regularization")
        print("="*60)
        print(f"Training mode: Offline (using replay data)")
        print(f"Data source: {self.config['data']['data_dir']}")
        if use_bc_init:
            print(f"BC pretrain: {self.config['experiment'].get('bc_pretrain_path', 'N/A')}")
        else:
            print(f"Initialization: Random (BC init disabled)")
        print(f"CQL alpha: {self.config['training'].get('cql_alpha', 0.0)}")
        print(f"BC weight: {self.config['training'].get('bc_weight', 0.0)}")
        print(f"Gradient clip: {self.config['training'].get('gradient_clip', 1.0)}")
        if self.resume_checkpoint:
            print(f"Resuming from: {self.resume_checkpoint}")
            print(f"Start epoch: {self.start_epoch}, Start step: {self.start_step}")
        print(f"Total epochs: {total_epochs}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        if steps_per_epoch:
            print(f"Steps per epoch: {steps_per_epoch}")
        print("="*60 + "\n")
        
        for epoch in range(self.start_epoch, total_epochs):
            epoch_losses = {
                'critic_loss': [],
                'bellman_loss': [],
                'cql_loss': [],
                'actor_loss': [],
                'sac_actor_loss': [],
                'bc_loss': [],
                'alpha': [],
                'entropy': []
            }
            
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{total_epochs}", 
                       total=steps_per_epoch if steps_per_epoch else None)
            
            epoch_step = 0
            for batch_data in pbar:
                obs, memory, actions, rewards, next_obs, next_memory, dones = batch_data
                
                batch = {
                    'observations': obs.to(self.device),
                    'memories': memory.to(self.device),
                    'actions': actions.to(self.device),
                    'rewards': rewards.to(self.device).float(),
                    'next_observations': next_obs.to(self.device),
                    'next_memories': next_memory.to(self.device),
                    'dones': dones.to(self.device).float()
                }
                
                metrics = self.agent.update(batch)
                
                for key in epoch_losses:
                    if key in metrics:
                        epoch_losses[key].append(metrics[key])
                
                global_step += 1
                epoch_step += 1
                
                if steps_per_epoch and epoch_step >= steps_per_epoch:
                    break
                
                if global_step % log_frequency == 0:
                    avg_critic_loss = np.mean(epoch_losses['critic_loss'][-100:]) if epoch_losses['critic_loss'] else 0
                    avg_actor_loss = np.mean(epoch_losses['actor_loss'][-100:]) if epoch_losses['actor_loss'] else 0
                    avg_alpha = np.mean(epoch_losses['alpha'][-100:]) if epoch_losses['alpha'] else 0
                    avg_entropy = np.mean(epoch_losses['entropy'][-100:]) if epoch_losses['entropy'] else 0
                    
                    pbar.set_postfix({
                        'c_loss': f'{avg_critic_loss:.3f}',
                        'a_loss': f'{avg_actor_loss:.3f}',
                        'alpha': f'{avg_alpha:.3f}',
                        'entropy': f'{avg_entropy:.2f}'
                    })
                    
                    if self.use_wandb:
                        log_dict = {
                            'critic_loss': avg_critic_loss,
                            'bellman_loss': np.mean(epoch_losses['bellman_loss'][-100:]) if epoch_losses['bellman_loss'] else 0,
                            'cql_loss': np.mean(epoch_losses['cql_loss'][-100:]) if epoch_losses['cql_loss'] else 0,
                            'actor_loss': avg_actor_loss,
                            'sac_actor_loss': np.mean(epoch_losses['sac_actor_loss'][-100:]) if epoch_losses['sac_actor_loss'] else 0,
                            'bc_loss': np.mean(epoch_losses['bc_loss'][-100:]) if epoch_losses['bc_loss'] else 0,
                            'alpha': avg_alpha,
                            'entropy': avg_entropy,
                            'epoch': epoch,
                            'step': global_step
                        }
                        wandb.log(log_dict, step=global_step)
            
            avg_epoch_metrics = {
                key: np.mean(values) if values else 0.0 
                for key, values in epoch_losses.items()
            }
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Critic Loss: {avg_epoch_metrics['critic_loss']:.4f}")
            print(f"  Actor Loss: {avg_epoch_metrics['actor_loss']:.4f}")
            print(f"  Alpha: {avg_epoch_metrics['alpha']:.4f}")
            print(f"  Entropy: {avg_epoch_metrics['entropy']:.4f}")
            
            if (epoch + 1) % save_frequency == 0:
                self.save_checkpoint(f"epoch_{epoch+1}", epoch=epoch+1, global_step=global_step)
        
        self.save_checkpoint('final', epoch=total_epochs, global_step=global_step)
        print("\n" + "="*60)
        print("Training completed!")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print("="*60)
    
    def save_checkpoint(self, name, epoch=None, global_step=None):
        checkpoint = {
            'name': name,
            'epoch': epoch,
            'global_step': global_step,
            'actor_state_dict': self.agent.actor.state_dict(),
            'critic_1_state_dict': self.agent.critic_1.state_dict(),
            'critic_2_state_dict': self.agent.critic_2.state_dict(),
            'critic_1_target_state_dict': self.agent.critic_1_target.state_dict(),
            'critic_2_target_state_dict': self.agent.critic_2_target.state_dict(),
            'actor_optimizer': self.agent.actor_optimizer.state_dict(),
            'critic_optimizer': self.agent.critic_optimizer.state_dict(),
            'alpha_optimizer': self.agent.alpha_optimizer.state_dict(),
            'log_alpha': self.agent.log_alpha.detach().cpu(),
        }
        
        save_path = self.checkpoint_dir / f'{name}.pt'
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint: {save_path}")
    
    def load_checkpoint(self):
        print(f"\nLoading checkpoint from: {self.resume_checkpoint}")
        self.start_epoch, self.start_step = self.agent.load(self.resume_checkpoint)
        print(f"Resuming from epoch {self.start_epoch}, step {self.start_step}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_offline_sac.yaml')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    trainer = OfflineSACTrainer(args.config, resume_checkpoint=args.resume)
    trainer.train()


if __name__ == '__main__':
    main()

