from gymnasium import spaces
from typing import TypedDict
import numpy as np


class Observation(TypedDict):
    p_boat: np.ndarray[2]           # m 
    theta_boat: np.ndarray[1]       # rad
    dt_p_boat: np.ndarray[2]        # m/s
    dt_theta_boat: np.ndarray[1]    # rad/s
    rpm_me: np.ndarray[1]           # rpm, meaning # 
    dt_rpm_me: np.ndarray[1]        # rpm/dt  
    theta_rudder: np.ndarray[1]     # rad
    dt_theta_rudder: np.ndarray[1]  # rad/s 
    N_thrusters: np.ndarray[2]      # Newton
    current: np.ndarray[2]          # angle clockwise south2north (deg), speed (m/s)
    wind: np.ndarray[2]             # angle clockwise north2south (deg), speed (m/s)
    wave: np.ndarray[2]             # angle clockwise north2south (deg), significant height (m)    


class Action(TypedDict):
    c_rpm_me: np.ndarray[1]         # rpm, meaning # 
    c_theta_rudder: np.ndarray[1]   # rad
    c_N_thrusters: np.ndarray[2]    # Newton


class ResetInfo(TypedDict):
    map_bounds: np.ndarray[2, 3]  # min, max


GymObservation = spaces.Dict({
    "p_boat": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
    "theta_boat": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
    "dt_p_boat": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
    "dt_theta_boat": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
    "rpm_me": spaces.Box(low=-200, high=200, shape=(1,), dtype=np.float32),
    "dt_rpm_me": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
    "theta_rudder": spaces.Box(low=-np.deg2rad(40), high=np.deg2rad(40), shape=(1,), dtype=np.float32),
    "dt_theta_rudder": spaces.Box(low=-np.deg2rad(4), high=np.deg2rad(4), shape=(1,), dtype=np.float32),
    "N_thrusters": spaces.Box(low=-100000, high=100000, shape=(2,), dtype=np.float32),
    "current": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
    "wind": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
    "wave": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
})

GymAction = spaces.Dict({
    "c_rpm_me": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
    "c_theta_rudder": spaces.Box(low=-np.deg2rad(40), high=np.deg2rad(40), shape=(1,), dtype=np.float32),
    "c_N_thrusters": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
})
