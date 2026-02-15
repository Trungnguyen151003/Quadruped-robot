# Quadruped Robot Locomotion in PyBullet (SpotDog) — TD3 vs DDPG

Simulation-based design and control of an 8-DoF/12-joint quadruped robot (SpotDog) in PyBullet.
This project trains a locomotion policy using **TD3** and provides **DDPG** as a baseline, evaluated on flat terrain and sloped terrains (0°, 5°, 9°).

## Highlights
- PyBullet simulation environment + URDF robot
- TD3 implementation (twin critics, delayed policy updates, target policy smoothing)
- Baseline: DDPG under the same training setup
- Evaluation on slope terrains: 0°, 5°, 9°
- Logs/plots: reward curves, roll/pitch stability, demo scripts

## Repository Structure
- Robot/ : URDF + robot assets
- utils/ : utilities (kinematics/terrain helpers if any)
- DDPG/ : DDPG implementation (baseline)
- train_TD3.py : training script
- demo.py : Training test and saving the result

## Demo TD3
### Flat terain 
https://www.youtube.com/watch?v=mleIhm7A_gQ

<img width="520" height="291" alt="Flat" src="https://github.com/user-attachments/assets/c7128ce0-b049-4ee7-af67-2ff9daed6911" />

### 5-degree slope
https://www.youtube.com/watch?v=WYjNU7DuYsE
 
<img width="568" height="310" alt="5degree" src="https://github.com/user-attachments/assets/07d007b5-6ee7-4cba-87d7-c34ebe349005" />

### 9-degree slope
https://www.youtube.com/watch?v=D4nhlst5EFM

<img width="603" height="373" alt="9degree" src="https://github.com/user-attachments/assets/62a96f42-52e5-409f-9d5d-f94b4e9e9b73" />

## Reward (TD3 vs DDPG)
### TD3 converges more stably than DDPG, especially on more challenging slopes.
- Flat terain
<img width="700" height="400" alt="TD3 vs DDPG (Flat)" src="https://github.com/user-attachments/assets/cbd0fec8-ae9c-40e4-a2f5-49432d556ea5" />

- 5-degree slope
<img width="700" height="400" alt="TD3 vs DDPG (5degree)" src="https://github.com/user-attachments/assets/53fb2b3c-3798-413e-bb3d-5ccccc52d5af" />

- 9-degree slope
<img width="700" height="400" alt="TD3 vs DDPG (9degree)" src="https://github.com/user-attachments/assets/be70747b-149c-4c8e-8b09-64aa59b7df00" />

## Limitations
- Simulation-only (PyBullet): real-world uncertainties are simplified, so direct hardware transfer isn’t guaranteed.
- Limited evaluation: tested mainly on flat ground and simple slopes (0°, 5°, 9°).
- Training constraints: compute budget and reward tuning limit extensive experiments and gait diversity.

## Future Work
- Sim-to-real: domain randomization + system identification to reduce the reality gap.
- Hardware deployment: integrate the policy with low-level control and safety constraints on a real robot.
- Robustness & gaits: harder terrains (uneven/stairs/obstacles) and more gait styles (pace/bound/gallop).

