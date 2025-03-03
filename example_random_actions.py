import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
import marine_gym
import cv2

env = gym.make('ShipEnv-v0', renderer=marine_gym.CV2DRenderer())
# env = RecordVideo(env, video_folder='./output/videos/')

env.reset(seed=10)

for i in range(100):
    act = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(act)
    if truncated:
        break
    # env.render()

    # Get the RGB frame
    frame = env.render()

    # Convert RGB to BGR for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Show the frame in OpenCV window
    cv2.imshow("Gymnasium Environment", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break
    input()

env.close()

 