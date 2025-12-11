from generals.agents import RandomAgent, ExpanderAgent
from generals.envs import PettingZooGenerals
import time

def print_grid(env):
    game = env.game
    channels = game.channels
    h, w = game.grid_dims
    
    # Map agents to symbols
    symbols = {}
    for i, agent in enumerate(game.agents):
        symbols[agent] = chr(ord('A') + i)

    print(f"\nTurn: {game.time}")
    print("  " + "".join([str(i%10) for i in range(w)]))
    
    for r in range(h):
        line = f"{r%10} "
        for c in range(w):
            char = '.'
            if channels.mountains[r, c]:
                char = '#'
            elif channels.cities[r, c]:
                char = '*' # Neutral city default
            
            # Check ownership
            owner = None
            for agent in game.agents:
                if channels.ownership[agent][r, c]:
                    owner = agent
                    break
            
            if owner:
                sym = symbols[owner]
                if channels.generals[r, c]:
                    char = sym # General
                elif channels.cities[r, c]:
                    char = sym # Owned city
                else:
                    char = sym.lower() # Land
            
            line += char
        print(line)

# Initialize agents
random = RandomAgent()
expander = ExpanderAgent()

# Names are used for the environment
agent_names = [random.id, expander.id]
# Store agents in a dictionary
agents = {
    random.id: random,
    expander.id: expander
}

# Create environment
env = PettingZooGenerals(agents=agent_names, render_mode=None)
observations, info = env.reset()

terminated = truncated = False
while not (terminated or truncated):
    actions = {}
    for agent in env.agents:
        # Ask agent for action
        actions[agent] = agents[agent].act(observations[agent])
    # All agents perform their actions
    observations, rewards, terminated, truncated, info = env.step(actions)
    print_grid(env)
    time.sleep(0.1)
