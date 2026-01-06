import numpy as np

from Robot import spot_pybullet_env


action = [0.0] * 12
action = np.array(action)
steps = 2000
total_reward = 0
robot = spot_pybullet_env.SpotEnv(render=True, on_rack=False, end_steps=steps)


for i in range(steps):
    state, reward, done, _ = robot.step(action)
    total_reward += reward
    robot._pybullet_client.resetDebugVisualizerCamera(3, 0, -89.99, robot.get_base_pos_and_orientation()[0])
print("Total reward: ", total_reward)
