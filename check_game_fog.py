from generals.core.game import Game
from generals.core.grid import Grid
import numpy as np

def check_game_fog():
    # A: Player 0
    # B: Player 1
    # C: City (neutral)
    grid_str = """
A...B
.....
..C..
.....
.....
""".strip()
    
    grid = Grid(grid_str)
    game = Game(grid, ["player_0", "player_1"])
    
    obs = game.agent_observation("player_0")
    tensor = obs.as_tensor()
    
    fog_cells = tensor[7]
    structures_in_fog = tensor[8]
    cities = tensor[2]
    
    # City at (2,2)
    r, c = 2, 2
    
    print(f"At (2,2) [Neutral City]:")
    print(f"  Fog: {fog_cells[r, c]}")
    print(f"  Structure in Fog: {structures_in_fog[r, c]}")
    print(f"  City (Visible): {cities[r, c]}")
    
    if structures_in_fog[r, c] > 0:
        print("WARNING: Unseen city is visible in 'structures_in_fog'!")
    else:
        print("Unseen city is hidden.")

if __name__ == "__main__":
    check_game_fog()
