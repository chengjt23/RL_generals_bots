
import numpy as np
from generals.core.grid import Grid
from generals.core.game import Game

# Create a simple grid
grid_str = """
. . . . .
. A . . .
. . # . .
. . . B .
. . . . .
"""
grid = Grid(grid_str.strip())
game = Game(grid, ["player_0", "player_1"])

# Get observation for player 0
obs = game.agent_observation("player_0")
tensor = obs.as_tensor()

print(f"Observation tensor shape: {tensor.shape}")

# Print non-zero elements for each channel to identify them
channel_names = [
    "Fog", "Obstacles (Fog)", "Cities", "Generals", "Friendly Units", "Enemy Units", 
    "Army Count", "Turn Number", "Unknown 1", "Unknown 2", "Unknown 3", "Unknown 4",
    "Unknown 5", "Unknown 6", "Unknown 7"
]

for i in range(tensor.shape[0]):
    print(f"\nChannel {i}:")
    print(tensor[i])
