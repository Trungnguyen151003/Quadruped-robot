import numpy as np
import torch
import os
import time
from Robot.spot_pybullet_env import SpotEnv
from td3_buffer import ReplayBuffer
from td3_agent import TD3
import matplotlib.pyplot as plt

if not os.path.exists("./11models"):
    os.makedirs("./11models")
if not os.path.exists('./11results'):
    os.makedirs('./11results')

env = SpotEnv(render=False, end_steps=4000, on_rack=False, imu_noise=True)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

policy = TD3(state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer()
episodes = 301   # số episode muốn train thêm
steps = 1000
batch_size = 256

start_timesteps = 10000
total_timesteps = 0

# ---- Cấu hình resume ----
resume = False  # True nếu muốn train tiếp, False nếu muốn train mới
resume_episode = 200  # Checkpoint đã lưu, ví dụ 1000 <=> td3_1000_actor.pth

if resume:
    policy.load(f"td3_{resume_episode}", "./1models")
    print(f"=> Loaded weights from ./1models/td3_{resume_episode}_actor.pth và critic.pth")
    start_episode = resume_episode + 1
    if os.path.exists('./1results/rewards.npy'):
        rewards = list(np.load('./1results/rewards.npy'))
    else:
        rewards = []
else:
    start_episode = 0
    rewards = []

for episode in range(episodes):
    state = env.reset()
    episode_reward = 0
    for step in range(steps):
        if len(replay_buffer.storage) < batch_size:
            action = env.action_space.sample()
        else:
            action = policy.select_action(np.array(state))
            action = (action + np.random.normal(0, 0.1, size=action_dim)).clip(env.action_space.low, env.action_space.high)
        new_state, reward, done, _ = env.step(action)
        replay_buffer.add((state, new_state, action, reward, float(done)))
        state = new_state
        episode_reward += reward

        if len(replay_buffer.storage) > batch_size:
            policy.train(replay_buffer, 1, batch_size)

        if done:
            break
    rewards.append(episode_reward)
    print(f"Episode: {episode}, Reward: {episode_reward}, Avg: {np.mean(rewards[-10:])}")
    if episode % 100 == 0:
        policy.save(f"td3_{episode}", "./11models")

    np.save('./11results/rewards.npy', rewards)  # auto save reward sau mỗi episode

# Tính moving average cho biểu đồ
window = 10
avg_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('TD3 Training Reward')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(avg_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward (Moving Average)')
plt.title('TD3 Training Average Reward')
plt.grid(True)

plt.tight_layout()
plt.show()
