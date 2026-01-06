import numpy as np
import torch
import matplotlib.pyplot as plt
from Robot.spot_pybullet_env import SpotEnv
from td3_agent import TD3

# Khởi tạo môi trường
env = SpotEnv(render=True, end_steps=4000, on_rack=False, imu_noise=False)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

policy = TD3(state_dim, action_dim, max_action)
policy.load('td3_300', './11models')   # Sửa tên file model cho đúng

# --- Chạy 1 episode và log roll, pitch mỗi bước ---
rolls = []
pitchs = []
rewards = []

state = env.reset()
total_reward = 0
for step in range(4000):
    action = policy.select_action(np.array(state))
    state, reward, done, _ = env.step(action)
    env._pybullet_client.resetDebugVisualizerCamera(2, 50, -20, env.get_base_pos_and_orientation()[0])
    # Lấy roll, pitch (radian)
    pos, ori = env.get_base_pos_and_orientation()
    rpy = env._pybullet_client.getEulerFromQuaternion(ori)
    rolls.append(np.degrees(rpy[0]))   # đổi sang độ cho dễ nhìn
    pitchs.append(np.degrees(rpy[1]))
    rewards.append(reward)
    total_reward += reward
    if done:
        break

print(f"Total reward: {total_reward}")

# --- Vẽ biểu đồ ---
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axs[0].plot(rolls, label="Roll (deg)")
axs[0].set_ylabel("Roll (°)")
axs[0].legend()

axs[1].plot(pitchs, label="Pitch (deg)", color='orange')
axs[1].set_ylabel("Pitch (°)")
axs[1].legend()

# axs[2].plot(rewards, label="Reward per step", color='green')
# axs[2].set_xlabel("Timestep")
# axs[2].set_ylabel("Reward")
# axs[2].legend()

plt.suptitle("Roll, Pitch in 1 episode")
plt.tight_layout()
plt.show()

