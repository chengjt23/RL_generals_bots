import sys
sys.path.append('..')

import os
import argparse
import torch
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
from agents.ppo_agent import PPOAgent
from agents.sota_agent import SOTAAgent
from generals.envs import PettingZooGenerals
import imageio

try:
    import pygame
    from PIL import Image
    RENDER_AVAILABLE = True
except ImportError:
    RENDER_AVAILABLE = False
    print("Warning: pygame or PIL not available, video recording disabled")


def compete(
    ppo_checkpoint: str = None,
    ppo_checkpoint2: str = None,
    bc_model_path: str = None,
    num_games: int = 100,
    verbose: bool = False,
    save_video: bool = False,
    video_dir: str = "videos",
    max_steps: int = 1000,
    mode: str = "ppo_vs_bc"
):
    """
    Competition between agents.
    
    Args:
        ppo_checkpoint: Path to PPO checkpoint (required for ppo_vs_bc or ppo_vs_ppo)
        ppo_checkpoint2: Path to second PPO checkpoint (required for ppo_vs_ppo)
        bc_model_path: Path to BC model (required for ppo_vs_bc)
        num_games: Number of games to play
        verbose: Print result for every game
        save_video: Save each game as MP4 video
        video_dir: Directory to save videos
        max_steps: Maximum steps per game
        mode: "ppo_vs_bc" or "ppo_vs_ppo"
    """
    # Validate arguments
    if mode == "ppo_vs_bc":
        if ppo_checkpoint is None or bc_model_path is None:
            raise ValueError("ppo_vs_bc mode requires both ppo_checkpoint and bc_model_path")
        title = "PPO Agent vs BC Model Competition"
        agent1_name = "PPO"
        agent2_name = "BC"
    elif mode == "ppo_vs_ppo":
        if ppo_checkpoint is None or ppo_checkpoint2 is None:
            raise ValueError("ppo_vs_ppo mode requires both ppo_checkpoint and ppo_checkpoint2")
        title = "PPO Agent vs PPO Agent Competition"
        agent1_name = "PPO1"
        agent2_name = "PPO2"
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'ppo_vs_bc' or 'ppo_vs_ppo'")
    
    print("\n" + "="*70)
    print(title)
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Video recording setup
    record_videos = save_video
    if record_videos and not RENDER_AVAILABLE:
        print("Warning: Cannot record video (pygame or PIL not installed)")
        record_videos = False
    videos_dir = Path(video_dir).resolve()
    if record_videos:
        videos_dir.mkdir(exist_ok=True)
    
    print(f"\nLoading models...")
    
    # Common config
    agent_config = {
        'grid_size': 24,
        'obs_channels': 15,
        'memory_channels': 20,
        'base_channels': 64
    }
    
    # Load agent 1 (PPO)
    agent1 = PPOAgent(
        sota_config=agent_config,
        id=agent1_name,
        grid_size=24,
        device=device,
        model_path=ppo_checkpoint,
        memory_channels=20
    )
    agent1.network.eval()
    print(f"  {agent1_name} Agent loaded from: {ppo_checkpoint}")
    
    # Load agent 2 (BC or PPO)
    if mode == "ppo_vs_bc":
        agent2 = SOTAAgent(
            sota_config=agent_config,
            id=agent2_name,
            grid_size=24,
            device=device,
            model_path=bc_model_path,
            memory_channels=20
        )
        print(f"  {agent2_name} Agent loaded from: {bc_model_path}")
    else:  # ppo_vs_ppo
        agent2 = PPOAgent(
            sota_config=agent_config,
            id=agent2_name,
            grid_size=24,
            device=device,
            model_path=ppo_checkpoint2,
            memory_channels=20
        )
        agent2.network.eval()
        print(f"  {agent2_name} Agent loaded from: {ppo_checkpoint2}")
    
    render_mode = "human" if record_videos else None
    if render_mode == "human":
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        import pygame
    
    env = PettingZooGenerals(agents=[agent1_name, agent2_name], render_mode=render_mode)
    
    agent1_wins = 0
    agent2_wins = 0
    agent1_rewards = []
    agent2_rewards = []
    game_lengths = []
    
    print(f"\nStarting {num_games} games...")
    if record_videos:
        print(f"Recording all games to directory: {videos_dir}")
    print("="*70)
    
    for game_num in range(num_games):
        obs_dict, info = env.reset()
        agent1.reset()
        agent2.reset()
        
        terminated = truncated = False
        agent1_episode_reward = 0
        agent2_episode_reward = 0
        step = 0
        
        recording = record_videos
        frames = [] if recording else None
        
        pbar = tqdm(total=max_steps, desc=f"Game {game_num + 1}/{num_games}", 
                   leave=False, disable=(not verbose and (game_num + 1) % 10 != 0))
        
        while not (terminated or truncated) and step < max_steps:
            agent1_action = agent1.act(obs_dict[agent1_name], deterministic=True)
            # SOTAAgent doesn't accept deterministic parameter, only PPOAgent does
            if mode == "ppo_vs_ppo":
                agent2_action = agent2.act(obs_dict[agent2_name], deterministic=True)
            else:
                agent2_action = agent2.act(obs_dict[agent2_name])
            actions_dict = {agent1_name: agent1_action, agent2_name: agent2_action}
            
            obs_dict, rewards_dict, terminated, truncated, info = env.step(actions_dict)
            agent1_episode_reward += rewards_dict[agent1_name]
            agent2_episode_reward += rewards_dict[agent2_name]
            step += 1
            
            pbar.update(1)
            pbar.set_postfix({
                f'{agent1_name}_r': f'{agent1_episode_reward:.1f}',
                f'{agent2_name}_r': f'{agent2_episode_reward:.1f}'
            })
            
            if recording:
                env.render()
                try:
                    renderer = env.gui._GUI__renderer
                    surface = renderer.screen
                    img_str = pygame.image.tostring(surface, 'RGB')
                    img = Image.frombytes('RGB', surface.get_size(), img_str)
                    frames.append(np.array(img))
                except (AttributeError, NameError):
                    if step == 1:
                        print("Warning: Could not capture frames for video")
                    recording = False
        
        pbar.close()
        
        game_lengths.append(step)
        agent1_rewards.append(agent1_episode_reward)
        agent2_rewards.append(agent2_episode_reward)
        
        if rewards_dict[agent1_name] > rewards_dict[agent2_name]:
            agent1_wins += 1
            result = f"{agent1_name} WIN"
        elif rewards_dict[agent2_name] > rewards_dict[agent1_name]:
            agent2_wins += 1
            result = f"{agent2_name} WIN"
        else:
            result = "DRAW"
        
        if verbose or (game_num + 1) % 10 == 0:
            print(f"Game {game_num + 1:3d}/{num_games}: {result:12s} | "
                  f"Steps: {step:4d} | "
                  f"{agent1_name} reward: {agent1_episode_reward:6.1f} | "
                  f"{agent2_name} reward: {agent2_episode_reward:6.1f}")
    
        env.close()
    
        if recording and frames:
            video_path = videos_dir / f"game_{game_num + 1}.mp4"
            try:
                imageio.mimsave(video_path, frames, fps=10, quality=8)
                if verbose or (game_num + 1) % 10 == 0:
                    print(f"Saved video for game {game_num + 1} to '{video_path}'")
            except Exception as e:
                print(f"✗ Failed to save video for game {game_num + 1}: {e}")
    
    total_games = num_games
    draws = total_games - agent1_wins - agent2_wins
    agent1_win_rate = agent1_wins / total_games * 100
    agent2_win_rate = agent2_wins / total_games * 100
    
    avg_agent1_reward = sum(agent1_rewards) / total_games
    avg_agent2_reward = sum(agent2_rewards) / total_games
    avg_game_length = sum(game_lengths) / total_games
    
    print("\n" + "="*70)
    print("Competition Results")
    print("="*70)
    print(f"\nTotal Games: {total_games}")
    print(f"\nWin Statistics:")
    print(f"  {agent1_name} Agent:   {agent1_wins:3d} wins  ({agent1_win_rate:5.1f}%)")
    print(f"  {agent2_name} Agent:   {agent2_wins:3d} wins  ({agent2_win_rate:5.1f}%)")
    print(f"  Draws:       {draws:3d}       ({draws/total_games*100:5.1f}%)")
    print(f"\nAverage Rewards:")
    print(f"  {agent1_name} Agent:   {avg_agent1_reward:7.2f}")
    print(f"  {agent2_name} Agent:   {avg_agent2_reward:7.2f}")
    print(f"  Difference:  {avg_agent1_reward - avg_agent2_reward:+7.2f} ({agent1_name} advantage)")
    print(f"\nAverage Game Length: {avg_game_length:.1f} steps")
    
    if agent1_win_rate > 55:
        print(f"\n🎉 {agent1_name} Agent dominates with {agent1_win_rate:.1f}% win rate!")
    elif agent1_win_rate > 50:
        print(f"\n✓ {agent1_name} Agent has a slight advantage with {agent1_win_rate:.1f}% win rate")
    elif agent1_win_rate > 45:
        print(f"\n⚖ Very close match! {agent1_name}: {agent1_win_rate:.1f}% vs {agent2_name}: {agent2_win_rate:.1f}%")
    else:
        print(f"\n⚠ {agent2_name} Agent is stronger with {agent2_win_rate:.1f}% win rate")
    
    print("="*70 + "\n")
    
    return {
        'agent1_wins': agent1_wins,
        'agent2_wins': agent2_wins,
        'draws': draws,
        'agent1_win_rate': agent1_win_rate,
        'agent2_win_rate': agent2_win_rate,
        'avg_agent1_reward': avg_agent1_reward,
        'avg_agent2_reward': avg_agent2_reward,
        'avg_game_length': avg_game_length,
        'mode': mode
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO Agent Competition (PPO vs BC or PPO vs PPO)')
    parser.add_argument('--ppo_checkpoint', type=str, default=None,
                       help='Path to first PPO checkpoint file (required)')
    parser.add_argument('--ppo_checkpoint2', type=str, default=None,
                       help='Path to second PPO checkpoint file (required for ppo_vs_ppo mode)')
    parser.add_argument('--bc_model', type=str, default=None,
                       help='Path to BC model file (required for ppo_vs_bc mode)')
    parser.add_argument('--mode', type=str, default='ppo_vs_bc', choices=['ppo_vs_bc', 'ppo_vs_ppo'],
                       help='Competition mode: ppo_vs_bc or ppo_vs_ppo (default: ppo_vs_bc)')
    parser.add_argument('--num_games', type=int, default=100,
                       help='Number of games to play (default: 100)')
    parser.add_argument('--max_steps', type=int, default=1000,
                       help='Maximum steps per game (default: 1000)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print result for every game')
    parser.add_argument('--save_video', action='store_true',
                       help='Save each game as MP4 video')
    parser.add_argument('--video_dir', type=str, default='videos',
                       help='Directory to save videos (default: videos)')
    args = parser.parse_args()
    
    compete(
        ppo_checkpoint=args.ppo_checkpoint,
        ppo_checkpoint2=args.ppo_checkpoint2,
        bc_model_path=args.bc_model,
        num_games=args.num_games,
        verbose=args.verbose,
        save_video=args.save_video,
        video_dir=args.video_dir,
        max_steps=args.max_steps,
        mode=args.mode
    )

