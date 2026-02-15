# Quadruped Robot Locomotion in PyBullet (SpotDog) — TD3 vs DDPG

Simulation-based design and control of an 8-DoF/12-joint quadruped robot (SpotDog) in PyBullet.
This project trains a locomotion policy using **TD3** and provides **DDPG** as a baseline, evaluated on flat terrain and sloped terrains (0°, 5°, 9°).

## Highlights
- PyBullet simulation environment + URDF robot
- TD3 implementation (twin critics, delayed policy updates, target policy smoothing)
- Baseline: DDPG under the same training setup
- Evaluation on slope terrains: 0°, 5°, 9°
- Logs/plots: reward curves, roll/pitch stability, demo scripts


