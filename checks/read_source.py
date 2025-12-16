
import inspect
from generals.core.observation import Observation

try:
    print(inspect.getsource(Observation.as_tensor))
except Exception as e:
    print(f"Could not get source: {e}")
