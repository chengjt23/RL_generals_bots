import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
import time
import multiprocessing
import torch, yaml
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add RL_generals_bots to path
sys.path.insert(0, "/root/cyh/RL_generals_bots")
# sys.path.insert(0, "/root/generals-bots")
sys.path.insert(0, "/root/shared-nvme/cyh/generals-bots")

# Add PyBot to path
# sys.path.append("/root/generals-bots/generals/agents/PyBot")
sys.path.append("/root/shared-nvme/cyh/generals-bots/generals/agents/PyBot")
from generals.agents import ExpanderAgent
from generals.envs import PettingZooGenerals
from generals.core.grid import GridFactory
from eval.models.ppo_1220_cyh import SOTAAgent, PPOAgent
from generals.agents.pybot_agent import PyBotAgent
from generals.agents.eklipz_agent import EklipZAgent
try:
    from example_randombot import RandomBot
    from example_pathfinderbot import PathFinderBot
    from example_flobot import FloBot
    from example_afkbot import AfkBot
except ImportError:
    print("Could not import PyBots. Make sure PyBot is in the correct path.")
    # Define dummy bots if import fails
    class RandomBot: pass
    class PathFinderBot: pass
    class FloBot: pass
    class AfkBot: pass

class ProgressBar:
    def __init__(self, total, width=50):
        self.total = total
        self.width = width
        self.current = 0
        self.start_time = time.time()

    def update(self, step=1):
        self.current += step
        percent = self.current / self.total
        bar_len = int(self.width * percent)
        bar = '=' * bar_len + '-' * (self.width - bar_len)
        
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.current) / rate if rate > 0 else 0
        
        sys.stdout.write(f'\r[{bar}] {percent:.1%} ({self.current}/{self.total}) - {elapsed:.1f}s elapsed - ETA: {remaining:.1f}s')
        sys.stdout.flush()

    def finish(self):
        sys.stdout.write('\n')

def run_single_game(game_id, bot_class, bot_name, ppo_model_path, ppo_config_path, sota_model_path, sota_config_path, max_steps):
    # Initialize agents inside the process to avoid pickling issues and ensure clean state
    try:
        torch.set_num_threads(1)
        if bot_class == EklipZAgent:
            opponent_agent = EklipZAgent()
            opponent_agent.id = bot_name
        elif bot_class == SOTAAgent:
            with open(sota_config_path, 'r') as f:
                sota_config = yaml.safe_load(f)
                sota_config = sota_config['model']
            opponent_agent = SOTAAgent(
                id=bot_name,
                sota_config=sota_config,
                grid_size=24,
                model_path=sota_model_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                memory_channels=sota_config['base_channels']
            )
        else:
            opponent_agent = PyBotAgent(bot_class, id=bot_name)

        with open(ppo_config_path, 'r') as f:
            ppo_config = yaml.safe_load(f)
            ppo_config = ppo_config['model']
        
        ppo_agent = PPOAgent(
            id="PPO",
            sota_config=ppo_config,
            grid_size=24,
            model_path=ppo_model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            memory_channels=ppo_config['base_channels']
        )

        # Reset agents internal state
        if hasattr(opponent_agent, 'reset'):
            opponent_agent.reset()
        if hasattr(ppo_agent, 'reset'):
            ppo_agent.reset()

        agents = {
            opponent_agent.id: opponent_agent,
            ppo_agent.id: ppo_agent
        }
        agent_names = [opponent_agent.id, ppo_agent.id]

        # Grid setup
        grid_factory = GridFactory(
            min_grid_dims=(24, 24),
            max_grid_dims=(24, 24)
        )
        grid_factory.padding = False

        # Environment
        env = PettingZooGenerals(
            agents=agent_names, 
            grid_factory=grid_factory, 
            render_mode=None,
            truncation=max_steps
        )

        observations, info = env.reset()
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            actions = {}
            for agent_id in env.agents:
                if agent_id in observations:
                    actions[agent_id] = agents[agent_id].act(observations[agent_id])
            
            observations, rewards, terminated, truncated, info = env.step(actions)

        winner = "Draw"
        if terminated:
            opponent_reward = rewards.get(opponent_agent.id, 0)
            ppo_reward = rewards.get(ppo_agent.id, 0)
            
            if opponent_reward > ppo_reward:
                winner = opponent_agent.id
            elif ppo_reward > opponent_reward:
                winner = "PPO"
        
        env.close()
        return winner
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

def compete():
    # Configuration
    NUM_GAMES = 10
    MAX_STEPS = 1000
    BATCH_SIZE = 4
    
    # PPO Model (Main Agent)
    PPO_MODEL_PATH = "/root/shared-nvme/cyh/RL_generals_bots/experiments/ppo_selfplay_ratio_20251220_164850/checkpoints/checkpoint_40.pt"
    PPO_CONFIG_PATH = "/root/shared-nvme/cyh/RL_generals_bots/experiments/ppo_selfplay_ratio_20251220_164850/config.yaml"

    # SOTA Model (Baseline Opponent)
    SOTA_MODEL_PATH = "/root/shared-nvme/cyh/RL_generals_bots/experiments/bc_new_mem_all_replays_20251218_014906/checkpoints/epoch_37_loss_1.7065.pt" 
    SOTA_CONFIG_PATH = "/root/shared-nvme/cyh/RL_generals_bots/experiments/bc_new_mem_all_replays_20251218_014906/config.yaml"
    
    bots_to_test = [
        ("RandomBot", RandomBot),
        ("PathFinderBot", PathFinderBot),
        ("FloBot", FloBot),
        ("Afk", AfkBot),
        ("EklipZ", EklipZAgent),
        ("SOTA", SOTAAgent)
    ]

    print(f"Starting competition: PPO vs [RandomBot, PathFinderBot, FloBot, AfkBot, EklipZ, SOTA]")
    print(f"Games per Bot: {NUM_GAMES}, Max Steps: {MAX_STEPS}, Parallel Workers: {BATCH_SIZE}")
    print(f"PPO Model: {PPO_MODEL_PATH}")
    print(f"SOTA Model (for baseline): {SOTA_MODEL_PATH}")
    
    all_results = []

    for bot_name, bot_class in bots_to_test:
        print(f"\nRunning matches against {bot_name}...")
        
        results = {
            bot_name: 0,
            "PPO": 0,
            "Draw": 0,
            "Error": 0
        }

        start_time = time.time()
        
        # Use ProcessPoolExecutor for parallel execution
        with ProcessPoolExecutor(max_workers=BATCH_SIZE) as executor:
            futures = [
                executor.submit(run_single_game, i, bot_class, bot_name, PPO_MODEL_PATH, PPO_CONFIG_PATH, SOTA_MODEL_PATH, SOTA_CONFIG_PATH, MAX_STEPS) 
                for i in range(NUM_GAMES)
            ]
            
            progress = ProgressBar(NUM_GAMES)
            
            for future in as_completed(futures):
                winner = future.result()
                if winner in results:
                    results[winner] += 1
                else:
                    results["Error"] += 1
                
                progress.update()
                
            progress.finish()

        end_time = time.time()
        duration = end_time - start_time
        
        results['duration'] = duration
        results['bot_name'] = bot_name

        bot_name = results['bot_name']
        ppo_wins = results['PPO']
        bot_wins = results[bot_name]
        draws = results['Draw']
        total = NUM_GAMES 
        
        ppo_win_rate = (ppo_wins / total) * 100
        
        print(f"{bot_name:<20} | {ppo_wins:<10} | {bot_wins:<10} | {draws:<10} | {ppo_win_rate:>13.1f}%")
        
        all_results.append(results)

    # Output results table
    print("\n" + "="*85)
    print(f"{'Bot Name':<20} | {'PPO Wins':<10} | {'Bot Wins':<10} | {'Draws':<10} | {'PPO Win Rate':<15}")
    print("-" * 85)
    
    for res in all_results:
        bot_name = res['bot_name']
        ppo_wins = res['PPO']
        bot_wins = res[bot_name]
        draws = res['Draw']
        total = NUM_GAMES 
        
        ppo_win_rate = (ppo_wins / total) * 100
        
        print(f"{bot_name:<20} | {ppo_wins:<10} | {bot_wins:<10} | {draws:<10} | {ppo_win_rate:>13.1f}%")
    print("="*85)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    compete()
