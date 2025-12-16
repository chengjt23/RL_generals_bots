
import numpy as np
from generals.core.grid import Grid
from generals.core.game import Game
from generals.core.observation import Observation

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

print("\nGround Truth:")
print("Armies:\n", obs.armies)
print("Generals:\n", obs.generals)
print("Cities:\n", obs.cities)
# print("Swamps:\n", obs.swamps)

# Print channel values summary
for i in range(tensor.shape[0]):
    print(f"\nChannel {i}: Min={tensor[i].min()}, Max={tensor[i].max()}, Mean={tensor[i].mean()}")
    if tensor[i].max() > 0:
        print(tensor[i])
