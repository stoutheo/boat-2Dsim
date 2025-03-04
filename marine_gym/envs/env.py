import gymnasium as gym
import numpy as np
from typing import Any, Tuple, Callable, Type, TypedDict, Union

from ..types import GymObservation, GymAction, Observation, Action, ResetInfo
from ..utils import ProfilingMeta, is_debugging_all
from ..abstracts import AbcRender


import marine_gym.simulators.fortran_based.fortran_sim as fortr_sim


class Metadata(TypedDict):
    render_modes: list[str]
    render_fps: float


class VesselEnv(gym.Env, metaclass=ProfilingMeta):
    NB_STEPS_PER_SECONDS: int  # Hz

    render_mode: str
    metadata: Metadata

    action_space = GymAction
    observation_space = GymObservation

    def reset(self, **kwargs) -> Tuple[Observation, ResetInfo]:
        return super().reset(**kwargs)

    def step(self, action: Action) -> Tuple[Observation, float, bool, Any]:
        return super().step(action)

    def render(self, draw_extra_fct: Callable[[AbcRender, np.ndarray, Observation], None] = None) -> np.ndarray:
        return super().render()

    def close(self) -> None:
        return super().close()



default_scenario_info = {
                        'map_bounds': np.array([[   -500., -500.,    0.], [ 500.,  500.,    1.]], dtype=np.float32),
                        'ref_path': None
                    }

def dummy_update(obs_dict):
    
    obs_dict['p_boat'][0] += 0.1
    obs_dict['p_boat'][1] += 15  
    obs_dict['theta_boat'][0] += np.deg2rad(2) 

    obs_dict['dt_p_boat'][0] = 0.5
    obs_dict['dt_p_boat'][1] = 1.0  
    obs_dict['dt_theta_boat'][0] += -np.deg2rad(1)/10

    obs_dict['rpm_me'][0] = 100

    obs_dict['theta_rudder'][0] += np.deg2rad(1)/10. 
    obs_dict['dt_theta_rudder'][0] = np.deg2rad(4)/10. 

    obs_dict['N_thrusters'][0] = 30000
    obs_dict['N_thrusters'][1] = 30000

    obs_dict['current'][0] = 45
    obs_dict['current'][1] = 2   

    obs_dict['wind'][0] = 30
    obs_dict['wind'][1] = 1

    obs_dict['wave'][0] = 100
    obs_dict['wave'][1] = 3

    return obs_dict



def fortran_sim_update(obs_dict, actions, simulator):
    
    # unwrap to match fortran simulator format
    init_state = np.array([ obs_dict['p_boat'][0],    obs_dict['p_boat'][1],    obs_dict['theta_boat'][0],
                            obs_dict['dt_p_boat'][0], obs_dict['dt_p_boat'][1], obs_dict['dt_theta_boat'][0],  
                            obs_dict['rpm_me'][0], np.rad2deg(obs_dict['theta_rudder'][0]), 
                            obs_dict['N_thrusters'][0], obs_dict['N_thrusters'][1]])

    controls = np.array([ actions['c_rpm_me'][0]/60., np.rad2deg(actions['c_theta_rudder'][0]), actions['c_N_thrusters'][0], actions['c_N_thrusters'][1] ])
    controls = np.array([ 100., -45., 30000., -30000.]) * 0.

    env_cond = np.array([obs_dict['current'][0], obs_dict['current'][1], obs_dict['wind'][0], obs_dict['wind'][1]])


    steps_of_horizon = 1
    dt = 1.

    state, _ = simulator.step(init_state, controls, env_cond, steps_of_horizon, dt, absolute_time = 0.0)

    # wrap again to match animation format

    obs_dict['p_boat'][0] = state[0,0]
    obs_dict['p_boat'][1] = state[1,0]
    obs_dict['theta_boat'][0] = state[2,0]

    obs_dict['dt_p_boat'][0] = state[3,0]
    obs_dict['dt_p_boat'][1] = state[4,0]  
    obs_dict['dt_theta_boat'][0] = state[5,0]

    obs_dict['rpm_me'][0] = state[6,0] * 60.

    obs_dict['theta_rudder'][0] = np.deg2rad(state[7,0])
    
    obs_dict['dt_theta_rudder'][0] = np.deg2rad((state[7,0] - init_state[7])/dt)

    obs_dict['N_thrusters'][0] = state[8,0]
    obs_dict['N_thrusters'][1] = state[9,0]
    
    obs_dict['current'][0] = env_cond[0]
    obs_dict['current'][1] = env_cond[1]

    obs_dict['wind'][0] = env_cond[2]
    obs_dict['wind'][1] = env_cond[3]

    obs_dict['wave'][0] = 0
    obs_dict['wave'][1] = 0

    return obs_dict


class ShipEnv(VesselEnv):
    NB_STEPS_PER_SECONDS = 1  # Hz

    def __init__(self, reward_fn: Callable[[Observation, Action, Observation], float] = lambda *_: 0, 
                 renderer: Union[AbcRender, None] = None, 
                 video_speed: float = 1, 
                 keep_sim_alive: bool = False, 
                 name='default', 
                 map_scale=1, 
                 stop_condition_fn: Callable[[Observation, Action, Observation], bool] = lambda *_: False):
        """Ship environment

        Args:
            reward_fn (Callable[[Observation, Action], float], optional): Use a custom reward function depending of your task. Defaults to lambda *_: 0.
            renderer (AbcRender, optional): Renderer instance to be used for rendering the environment, look at sailboat_gym/renderers folder for more information. Defaults to None.
            video_speed (float, optional): Speed of the video recording. Defaults to 1.
            keep_sim_alive (bool, optional): Keep the simulation running even after the program exits. Defaults to False.
            name ([type], optional): Name of the simulation, required to run multiples environment on same machine.. Defaults to 'default'.
            map_scale (int, optional): Scale of the map, used to scale the map in the renderer. Defaults to 1.
        """
        super().__init__()

        # IMPORTANT: The following variables are required by the gymnasium API
        self.render_mode = renderer.get_render_mode() if renderer else None
        self.metadata = {
            'render_modes': renderer.get_render_modes() if renderer else [],
            'render_fps': float(video_speed * self.NB_STEPS_PER_SECONDS),
        }

        def direction_generator(std=1.):
            direction = np.random.normal(0, std, 2)

            def generate_direction(step_idx):
                return direction
            return generate_direction

        self.name = name
        self.reward_fn = reward_fn
        self.stop_condition_fn = stop_condition_fn
        self.renderer = renderer
        self.obs = None
        self.wind_generator_fn = wind_generator_fn if wind_generator_fn \
            else direction_generator()
        self.water_generator_fn = water_generator_fn if water_generator_fn \
            else direction_generator(0.01)
        self.wave_generator_fn = wave_generator_fn if wave_generator_fn \
            else direction_generator(0.01)
        self.map_scale = map_scale
        self.keep_sim_alive = keep_sim_alive
        self.step_idx = 0

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        if seed is not None:
            np.random.seed(seed)
        self.step_idx = 0

        wind = self.wind_generator_fn(self.step_idx)
        water = self.water_generator_fn(self.step_idx)
        wave = self.wave_generator_fn(self.step_idx)

        # replace this 
        # self.obs, info = self.sim.reset(wind, water, self.NB_STEPS_PER_SECONDS)
        self.obs = zero_values
        info = env_info
        
        # setup the renderer, its needed to know the min/max position of the boat
        if self.renderer:
            self.renderer.setup(info['map_bounds'] * self.map_scale, info['ref_path'])

        if is_debugging_all():
            print('\nResetting environment:')
            print(f'  -> Wind: {wind}')
            print(f'  -> Water: {water}')
            print(f'  -> frequency: {self.NB_STEPS_PER_SECONDS} Hz')
            print(f'  <- Obs: {self.obs}')
            print(f'  <- Info: {info}')

        return self.obs, info

    def step(self, action: Action):
        assert self.obs is not None, 'Please call reset before step'

        self.step_idx += 1

        # replace this 
        next_obs = fortran_sim_update(self.obs, action, self.sim) 

        # could implement this to terminate if the end of the reference path has been reached
        terminated = False

        # set animation scenario
        info = {}

        # could implement this to return the metric of the tracking task 
        reward = self.reward_fn(self.obs, action, next_obs)

        truncated = self.stop_condition_fn(self.obs, action, next_obs)

        # update observations
        self.obs = next_obs

        if is_debugging_all():
            print('\nStepping environment:')
            print(f'  -> Action: {action}')
            print(f'  <- Obs: {self.obs}')
            print(f'  <- Reward: {reward}')
            print(f'  <- Terminated: {terminated}')

        return self.obs, reward, terminated, truncated, info

    def render(self):
        assert self.renderer, 'No renderer'
        assert self.obs is not None, 'Please call reset before render'
        return self.renderer.render(self.obs)

    def close(self):
        self.obs = None
