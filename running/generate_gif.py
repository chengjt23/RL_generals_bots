import os
import sys

# Add RL_generals_bots to path to import SOTAAgent
sys.path.append("/root/RL_generals_bots")

# Set dummy video driver before importing pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
from PIL import Image
from generals.agents import RandomAgent, ExpanderAgent
from generals.envs import PettingZooGenerals
from generals.core.grid import GridFactory
from agents.sota_agent import SOTAAgent

def generate_gif():
    # Initialize agents
    # random = RandomAgent()
    expander = ExpanderAgent()
    
    model_path = "/root/RL_generals_bots/experiments/behavior_cloning_20251209_202310/checkpoints/epoch_14_loss_2.5193.pt"
    sota = SOTAAgent(
        id="SOTA", 
        grid_size=24, 
        model_path=model_path,
        memory_channels=0
    )

    # Names are used for the environment
    agent_names = [sota.id, expander.id]
    agents = {
        sota.id: sota,
        expander.id: expander
    }

    # Create GridFactory with fixed size 24x24
    grid_factory = GridFactory(
        min_grid_dims=(24, 24),
        max_grid_dims=(24, 24)
    )
    # Manually set padding attribute to avoid AttributeError in PettingZooGenerals
    grid_factory.padding = False

    # Create environment with human render mode to enable GUI/Renderer
    env = PettingZooGenerals(agents=agent_names, grid_factory=grid_factory, render_mode="human")
    observations, info = env.reset()

    frames = []
    max_frames = 400  # Limit frames to keep GIF size manageable
    
    print("Starting simulation and capturing frames...")
    
    terminated = truncated = False
    frame_count = 0
    
    while not (terminated or truncated) and frame_count < max_frames:
        actions = {}
        for agent in env.agents:
            actions[agent] = agents[agent].act(observations[agent])
            
        observations, rewards, terminated, truncated, info = env.step(actions)
        env.render()
        
        # Capture frame
        # Access the renderer's screen surface
        # Note: accessing private member _GUI__renderer
        try:
            renderer = env.gui._GUI__renderer
            surface = renderer.screen
            
            # Convert pygame surface to PIL Image
            # pygame uses (width, height), PIL uses (width, height)
            # tostring 'RGB' works
            img_str = pygame.image.tostring(surface, 'RGB')
            img = Image.frombytes('RGB', surface.get_size(), img_str)
            frames.append(img)
            
        except AttributeError:
            print("Could not access renderer. Make sure render_mode='human' is set.")
            break
            
        frame_count += 1
        if frame_count % 20 == 0:
            print(f"Captured {frame_count} frames...")

    env.close()
    
    if frames:
        print(f"Saving GIF with {len(frames)} frames to 'gameplay.gif'...")
        # Save as GIF
        frames[0].save(
            'gameplay.gif',
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=100, # 100ms per frame = 10fps
            loop=0
        )
        print("Done! Saved to /root/generals-bots/examples/gameplay.gif")
    else:
        print("No frames captured.")

if __name__ == "__main__":
    generate_gif()
