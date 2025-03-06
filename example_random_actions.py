import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
import marine_gym
import cv2
import numpy as np

env = gym.make('ShipEnv-v0', renderer=marine_gym.CV2DRenderer())
env = RecordVideo(env, video_folder='./output/videos/')



route_info = {
            'map_bounds': np.array([[   -500., -500.,    0.], [ 500.,  500.,    1.]], dtype=np.float32),
            'ref_path': np.array([
                                [-450, -300], 
                                [-350, -100], 
                                [-200, 50], 
                                [0, 200], 
                                [150, 300], 
                                [300, 400], 
                                [350, 200], 
                                [200, 50], 
                                [50, -200], 
                                [-100, -400]
                            ])
            }


reset_options = {
                'scenario_info': route_info,
                'initial_state': None,
                'wind_generator_fn': None,
                'current_generator_fn': None,
                'wave_generator_fn': None,
                }


for ii in range(1):

    env.reset(seed=ii, options = reset_options)

    for i in range(1000):
        act = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(act)
        if truncated:
            break

        # Get the RGB frame
        frame = env.render()

        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Show the frame in OpenCV window
        cv2.imshow("Gymnasium Environment", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    # input()

env.close()

 

