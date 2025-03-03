from gymnasium import spaces
from typing import TypedDict
import numpy as np


class Observation(TypedDict):
    p_boat: np.ndarray[2]
    theta_boat: np.ndarray[1]
    dt_p_boat: np.ndarray[2]
    dt_theta_boat: np.ndarray[1]
    rpm_me: np.ndarray[1]
    dt_rpm_me: np.ndarray[1]
    theta_rudder: np.ndarray[1]
    dt_theta_rudder: np.ndarray[1]
    N_thrusters: np.ndarray[2]
    wind: np.ndarray[2]
    water: np.ndarray[2]
    wave: np.ndarray[2]


class Action(TypedDict):
    c_rpm_me: np.ndarray[1]
    c_theta_rudder: np.ndarray[1]
    c_N_thrusters: np.ndarray[2]


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
    "wind": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
    "water": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
    "wave": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
})

GymAction = spaces.Dict({
    "c_rpm_me": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
    "c_theta_rudder": spaces.Box(low=-np.deg2rad(40), high=np.deg2rad(40), shape=(1,), dtype=np.float32),
    "c_N_thrusters": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
})
