import sys
sys.path.append('..')

import os
import argparse
import torch
import yaml
from pathlib import Path
from tqdm import tqdm
from agents.sac_agent import SACAgent
from agents.sota_agent import SOTAAgent
from generals.envs import PettingZooGenerals

try:
    import pygame
    from PIL import Image
    GIF_AVAILABLE = True
except ImportError:
    GIF_AVAILABLE = False
    print("Warning: pygame or PIL not available, GIF recording disabled")


def compete(sac_checkpoint: str, bc_model_path: str, num_games: int = 100, verbose: bool = False, save_gif: bool = False, gif_path: str = "first_game.gif", max_steps: int = 1000):
    print("\n" + "="*70)
    print("SAC Agent vs BC Model Competition")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if save_gif and not GIF_AVAILABLE:
        print("Warning: Cannot save GIF (pygame or PIL not installed)")
        save_gif = False
    
    print(f"\nLoading models...")
    sac_agent = SACAgent(
        id="SAC",
        grid_size=24,
        device=device,
        model_path=sac_checkpoint,
        memory_channels=20
    )
    sac_agent.actor.eval()
    print(f"  SAC Agent loaded from: {sac_checkpoint}")
    
    bc_config = {
        'grid_size': 24,
        'obs_channels': 15,
        'memory_channels': 20,
        'base_channels': 64
    }
    bc_agent = SOTAAgent(
        sota_config=bc_config,
        id="BC",
        grid_size=24,
        device=device,
        model_path=bc_model_path,
        memory_channels=20
    )
    print(f"  BC Agent loaded from: {bc_model_path}")
    
    render_mode = "human" if (save_gif and GIF_AVAILABLE) else None
    if render_mode == "human":
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        import pygame
    
    env = PettingZooGenerals(agents=["SAC", "BC"], render_mode=render_mode)
    
    sac_wins = 0
    bc_wins = 0
    sac_rewards = []
    bc_rewards = []
    game_lengths = []
    
    frames = []
    
    print(f"\nStarting {num_games} games...")
    if save_gif:
        print(f"Recording first game to {gif_path}...")
    print("="*70)
    
    for game_num in range(num_games):
        obs_dict, info = env.reset()
        sac_agent.reset()
        bc_agent.reset()
        
        terminated = truncated = False
        sac_episode_reward = 0
        bc_episode_reward = 0
        step = 0
        
        recording = save_gif and game_num == 0
        
        pbar = tqdm(total=max_steps, desc=f"Game {game_num + 1}/{num_games}", 
                   leave=False, disable=(not verbose and (game_num + 1) % 10 != 0))
        
        while not (terminated or truncated) and step < max_steps:
            # print(step)
            sac_action = sac_agent.act(obs_dict["SAC"], deterministic=True)
            bc_action = bc_agent.act(obs_dict["BC"])
            actions_dict = {"SAC": sac_action, "BC": bc_action}
            
            obs_dict, rewards_dict, terminated, truncated, info = env.step(actions_dict)
            sac_episode_reward += rewards_dict["SAC"]
            bc_episode_reward += rewards_dict["BC"]
            step += 1
            
            pbar.update(1)
            pbar.set_postfix({
                'SAC_r': f'{sac_episode_reward:.1f}',
                'BC_r': f'{bc_episode_reward:.1f}'
            })
            
            if recording:
                env.render()
                try:
                    renderer = env.gui._GUI__renderer
                    surface = renderer.screen
                    img_str = pygame.image.tostring(surface, 'RGB')
                    img = Image.frombytes('RGB', surface.get_size(), img_str)
                    frames.append(img)
                except (AttributeError, NameError):
                    if step == 1:
                        print("Warning: Could not capture frames for GIF")
                    recording = False
        
        pbar.close()
        
        game_lengths.append(step)
        sac_rewards.append(sac_episode_reward)
        bc_rewards.append(bc_episode_reward)
        
        if rewards_dict["SAC"] > rewards_dict["BC"]:
            sac_wins += 1
            result = "SAC WIN"
        elif rewards_dict["BC"] > rewards_dict["SAC"]:
            bc_wins += 1
            result = "BC WIN"
        else:
            result = "DRAW"
        
        if verbose or (game_num + 1) % 10 == 0:
            print(f"Game {game_num + 1:3d}/{num_games}: {result:8s} | "
                  f"Steps: {step:4d} | "
                  f"SAC reward: {sac_episode_reward:6.1f} | "
                  f"BC reward: {bc_episode_reward:6.1f}")
    
    env.close()
    
    if frames and save_gif:
        print(f"\nSaving GIF with {len(frames)} frames to '{gif_path}'...")
        try:
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=100,
                loop=0
            )
            print(f"✓ GIF saved successfully to {gif_path}")
        except Exception as e:
            print(f"✗ Failed to save GIF: {e}")
    
    total_games = num_games
    draws = total_games - sac_wins - bc_wins
    sac_win_rate = sac_wins / total_games * 100
    bc_win_rate = bc_wins / total_games * 100
    
    avg_sac_reward = sum(sac_rewards) / total_games
    avg_bc_reward = sum(bc_rewards) / total_games
    avg_game_length = sum(game_lengths) / total_games
    
    print("\n" + "="*70)
    print("Competition Results")
    print("="*70)
    print(f"\nTotal Games: {total_games}")
    print(f"\nWin Statistics:")
    print(f"  SAC Agent:   {sac_wins:3d} wins  ({sac_win_rate:5.1f}%)")
    print(f"  BC Model:    {bc_wins:3d} wins  ({bc_win_rate:5.1f}%)")
    print(f"  Draws:       {draws:3d}       ({draws/total_games*100:5.1f}%)")
    print(f"\nAverage Rewards:")
    print(f"  SAC Agent:   {avg_sac_reward:7.2f}")
    print(f"  BC Model:    {avg_bc_reward:7.2f}")
    print(f"  Difference:  {avg_sac_reward - avg_bc_reward:+7.2f} (SAC advantage)")
    print(f"\nAverage Game Length: {avg_game_length:.1f} steps")
    
    if sac_win_rate > 55:
        print(f"\n🎉 SAC Agent dominates with {sac_win_rate:.1f}% win rate!")
    elif sac_win_rate > 50:
        print(f"\n✓ SAC Agent has a slight advantage with {sac_win_rate:.1f}% win rate")
    elif sac_win_rate > 45:
        print(f"\n⚖ Very close match! SAC: {sac_win_rate:.1f}% vs BC: {bc_win_rate:.1f}%")
    else:
        print(f"\n⚠ BC Model is stronger with {bc_win_rate:.1f}% win rate")
    
    print("="*70 + "\n")
    
    return {
        'sac_wins': sac_wins,
        'bc_wins': bc_wins,
        'draws': draws,
        'sac_win_rate': sac_win_rate,
        'bc_win_rate': bc_win_rate,
        'avg_sac_reward': avg_sac_reward,
        'avg_bc_reward': avg_bc_reward,
        'avg_game_length': avg_game_length
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAC Agent vs BC Model Competition')
    parser.add_argument('--sac_checkpoint', type=str, required=True, 
                       help='Path to SAC checkpoint file')
    parser.add_argument('--bc_model', type=str, required=True,
                       help='Path to BC model file')
    parser.add_argument('--num_games', type=int, default=100,
                       help='Number of games to play (default: 100)')
    parser.add_argument('--max_steps', type=int, default=1000,
                       help='Maximum steps per game (default: 1000)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print result for every game')
    parser.add_argument('--save_gif', action='store_true',
                       help='Save first game as GIF')
    parser.add_argument('--gif_path', type=str, default='first_game.gif',
                       help='Path to save GIF file (default: first_game.gif)')
    args = parser.parse_args()
    
    compete(
        sac_checkpoint=args.sac_checkpoint,
        bc_model_path=args.bc_model,
        num_games=args.num_games,
        verbose=args.verbose,
        save_gif=args.save_gif,
        gif_path=args.gif_path,
        max_steps=args.max_steps
    )

