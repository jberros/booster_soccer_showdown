"""
Teleoperate T1 robot in a gymnasium environment using a keyboard.
"""

import time
import argparse
import sai_mujoco  # noqa: F401
import gymnasium as gym
from se3_keyboard import Se3Keyboard
from t1_utils import LowerT1JoyStick

parser = argparse.ArgumentParser("Teleoperate T1 robot in a gymnasium environment.")
parser.add_argument(
    "--env", type=str, default="LowerT1GoaliePenaltyKick-v0", help="The environment to teleoperate."
)
parser.add_argument(
    "--max_fr", type=float, default=20.0, help="Maximum frame rate in Hz."
)

args = parser.parse_args()

env = gym.make(args.env, render_mode="human", control_freq=500)
t1_utils = LowerT1JoyStick(env.unwrapped)

# Initialize the T1 SE3 keyboard controller with the viewer
keyboard_controller = Se3Keyboard(
        renderer=env.unwrapped.mujoco_renderer
    )

# Set the reset environment callback
keyboard_controller.set_reset_env_callback(env.reset)

# Print keyboard control instructions
print("\nKeyboard Controls:")
print(keyboard_controller)

# Frame rate control
frame_time = 1.0 / args.max_fr
last_frame_time = time.time()

# Main teleoperation loop
episode_count = 0
while True:
    # Reset environment for new episode
    terminated = truncated = False
    obs, info = env.reset()
    rewards = []
    episode_count += 1

    print(f"\nStarting episode {episode_count}")

    # Episode loop
    while not (terminated):
        # Get keyboard input and apply it directly to the environment
        command = keyboard_controller.advance()
        updated_obs = t1_utils.get_obs(command)
        
        ctrl = t1_utils.get_actions(updated_obs)
        obs, reward, terminated, truncated, info = env.step(ctrl)
        rewards.append(reward)
        if terminated:
            break

    # Print episode result
    if info.get("success", False):
        print(f"Episode {episode_count} completed successfully!")
    else:
        print(f"Episode {episode_count} completed without success")