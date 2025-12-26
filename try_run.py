from generals.envs import PettingZooGenerals
from agents.sota_agent import SOTAAgent

env = PettingZooGenerals(agents=["SOTA", "Random"])
agent = SOTAAgent(id="SOTA", grid_size=24, 
                  model_path="experiments/behavior_cloning_20251209_202310/checkpoints/epoch_14_loss_2.5193.pt",
                  memory_channels=0)

obs_dict, info = env.reset()
obs_for_agent = obs_dict["SOTA"]
action = agent.act(obs_for_agent)
# 将 action 转换为环境需要的格式并 step