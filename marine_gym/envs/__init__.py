from gymnasium.envs.registration import register

from .env import ShipEnv

env_by_name = {
    'ShipEnv-v0': ShipEnv,
}

for name, env in env_by_name.items():
    register(
        id=name,
        entry_point=f'marine_gym.envs:{env.__name__}',
    )
