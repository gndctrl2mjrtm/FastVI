from VI import fast_value_iteration
from frozen_lake import *


# MAPS["new_env"] = generate_random_map(size=20)
env = FrozenLakeEnv(desc = generate_random_map(size=20))
print(env.reset())
print(env.reset().shape)