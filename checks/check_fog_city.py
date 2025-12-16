
import numpy as np
from generals.core.grid import Grid
from generals.core.game import Game

# Create a grid with a city and a general
# Player A is at (1,1)
# Neutral City is at (1,3) - initially visible? No, range is small.
# Let's put it far away.
grid_str = """
. . . . . .
. A . . B .
. . . . . .
. . . . . .
. . . . 4 .
. . . . . .
"""
# '4' in map string usually means neutral city with some troops? 
# In generals.io map format:
# Digits usually represent neutral cities. 
# Let's check dataloader._create_initial_grid to see how it parses.
# It maps city_value to char.
# But here we are using generals-bots Game directly.

# Let's try to construct a grid where we can control visibility.
# We will manually step the game.

grid = Grid(grid_str.strip())
game = Game(grid, ["player_0", "player_1"])

# Get initial observation
obs = game.agent_observation("player_0")
tensor = obs.as_tensor()

# Channel mapping (based on previous knowledge/inspection):
# 0: Fog (1 if fog, 0 if visible) - Wait, usually it's the other way or specific.
# Let's check the values.

print("Shape:", tensor.shape)

# Find the city location
# In the grid string, '4' is at row 4, col 4 (0-indexed).
r, c = 4, 4

print(f"Checking cell ({r}, {c}) which should be a neutral city.")

# Check Fog channel (usually 0)
print(f"Channel 0 (Fog?) at ({r},{c}): {tensor[0, r, c]}")

# Check City channel (usually 2)
print(f"Channel 2 (Cities) at ({r},{c}): {tensor[2, r, c]}")

# Check Army channel (usually 6 or similar, let's print all non-zero channels at r,c)
print(f"Values at ({r},{c}):")
for i in range(tensor.shape[0]):
    val = tensor[i, r, c]
    if val != 0:
        print(f"  Ch {i}: {val}")

