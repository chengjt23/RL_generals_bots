
import numpy as np

class MockObservation:
    def __init__(self):
        self.tensor = np.ones((15, 24, 24), dtype=np.float32) * 100.0
        
    def as_tensor(self):
        return self.tensor
        
    def pad_observation(self, pad_to):
        pass

obs = MockObservation()
obs_np = obs.as_tensor().astype(np.float32)

print(f"Original: {obs.tensor[0,0,0]}")
print(f"Copy: {obs_np[0,0,0]}")

obs_np[0] *= 0.5

print(f"Original after mod: {obs.tensor[0,0,0]}")
print(f"Copy after mod: {obs_np[0,0,0]}")
