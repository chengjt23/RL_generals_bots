import torch
import numpy as np
from agents.sota_agent import SOTAAgent
from agents.sac_agent import SACAgent
from generals.envs import PettingZooGenerals
from generals.agents import RandomAgent
from tqdm import tqdm


def evaluate_agent(agent, num_games=20, opponent="RandomAgent"):
    env = PettingZooGenerals(agents=[agent.id, opponent], render_mode=None)
    opponent_agent = RandomAgent()
    
    wins = 0
    total_rewards = 0
    episode_lengths = []
    
    for game_num in tqdm(range(num_games), desc=f"Evaluating {agent.id}"):
        obs_dict, info = env.reset()
        agent.reset()
        terminated = truncated = False
        episode_reward = 0
        step = 0
        
        while not (terminated or truncated) and step < 500:
            if isinstance(agent, SACAgent):
                agent_action = agent.act(obs_dict[agent.id], deterministic=True)
            else:
                agent_action = agent.act(obs_dict[agent.id])
            
            opponent_action = opponent_agent.act(obs_dict[opponent])
            actions_dict = {agent.id: agent_action, opponent: opponent_action}
            
            obs_dict, rewards_dict, terminated, truncated, info = env.step(actions_dict)
            episode_reward += rewards_dict[agent.id]
            step += 1
        
        if rewards_dict[agent.id] > rewards_dict[opponent]:
            wins += 1
        
        total_rewards += episode_reward
        episode_lengths.append(step)
    
    env.close()
    
    return {
        'win_rate': wins / num_games,
        'avg_reward': total_rewards / num_games,
        'avg_episode_length': np.mean(episode_lengths),
        'std_reward': np.std([total_rewards / num_games]),
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--bc_model', type=str, required=True, help='BC模型路径')
    parser.add_argument('--sac_checkpoint', type=str, default=None, help='SAC检查点路径（可选）')
    parser.add_argument('--num_games', type=int, default=20, help='评估游戏数量')
    parser.add_argument('--opponent', type=str, default='RandomAgent', help='对手类型')
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("智能体评估对比")
    print("=" * 70)
    
    print("\n1. 加载BC Agent...")
    bc_agent = SOTAAgent(
        id="BC",
        grid_size=24,
        device=device,
        model_path=args.bc_model,
        memory_channels=18
    )
    
    print(f"   ✓ BC模型加载成功: {args.bc_model}")
    
    print("\n2. 评估BC Agent...")
    bc_results = evaluate_agent(bc_agent, num_games=args.num_games, opponent=args.opponent)
    
    print("\n3. 加载SAC Agent...")
    sac_agent = SACAgent(
        id="SAC",
        grid_size=24,
        device=device,
        bc_model_path=args.bc_model,
        memory_channels=18
    )
    
    if args.sac_checkpoint is not None:
        checkpoint = torch.load(args.sac_checkpoint, map_location=device, weights_only=False)
        sac_agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        print(f"   ✓ SAC检查点加载成功: {args.sac_checkpoint}")
    else:
        print("   ⚠ 未提供SAC检查点，使用BC初始化的权重")
    
    print("\n4. 评估SAC Agent...")
    sac_results = evaluate_agent(sac_agent, num_games=args.num_games, opponent=args.opponent)
    
    print("\n" + "=" * 70)
    print("评估结果")
    print("=" * 70)
    
    print(f"\nBC Agent (行为克隆):")
    print(f"  胜率: {bc_results['win_rate']*100:.1f}%")
    print(f"  平均奖励: {bc_results['avg_reward']:.2f}")
    print(f"  平均步数: {bc_results['avg_episode_length']:.0f}")
    
    print(f"\nSAC Agent (软演员-评论家):")
    print(f"  胜率: {sac_results['win_rate']*100:.1f}%")
    print(f"  平均奖励: {sac_results['avg_reward']:.2f}")
    print(f"  平均步数: {sac_results['avg_episode_length']:.0f}")
    
    print("\n提升:")
    win_rate_improvement = (sac_results['win_rate'] - bc_results['win_rate']) * 100
    reward_improvement = sac_results['avg_reward'] - bc_results['avg_reward']
    
    print(f"  胜率提升: {win_rate_improvement:+.1f}%")
    print(f"  奖励提升: {reward_improvement:+.2f}")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()

