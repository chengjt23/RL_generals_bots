
import numpy as np
from generals.core.grid import Grid
from generals.core.game import Game

grid_str = """
. . . . .
. A . . B
. . . . .
"""
grid = Grid(grid_str.strip())
game = Game(grid, ["player_0", "player_1"])

# Hack to set armies
# Access internal grid?
# game.grid[1,1].army = 50 ? No.
# game.map.grid[1][1].army = 50 ?

# Let's try to find where army is stored.
# Or just step 50 times? No.

# Let's look at generals/core/game.py source if I could.
# But I can't.

# Let's try to infer from `dataloader.py`.
# `dataloader.py` uses:
# 'cities': tensor[2]
# 'generals': tensor[3]
# 'owned_cells': tensor[4]
# 'opponent_cells': tensor[5]

# If I look at `inspect_obs_2.py` (which I didn't run successfully because of numpy), I might see names.
# But I can run `check_army_channel.py`.

# Let's try to print the whole tensor for (1,1) if I can change the army.
# I can't easily change army without stepping.
# But I can step.
# If I step, armies grow on generals.
# Let's step 10 times.
for _ in range(10):
    game.step({"player_0": [1,0,0,0,0], "player_1": [1,0,0,0,0]}) # Pass

obs = game.agent_observation("player_0")
print("obs.armies shape:", obs.armies.shape)
print("obs.armies at (1,1):", obs.armies[1,1])


