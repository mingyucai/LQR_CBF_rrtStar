## Introduction

Control Barrier Functions (CBF) are a powerful tool for designing safety-critical controllers and motion planners. The safety requirements are encoded as a continuously differentiable function that maps from state variables to a real value, in which the sign of its output determines whether safety is violated. In practice, the CBFs can be used to enforce safety by imposing itself as a constraint in a Quadratic Program (QP) solved point-wise in time. However, this approach costs computational resources and could lead to infeasibility in solving the QP. 

In this project, we propose a novel motion planning framework that combines sampling-based methods with Linear Quadratic Regulator (LQR) and CBFs. Our approach does not require solving the QPs for control synthesis and avoids explicit collision checking during samplings. Instead, it uses LQR to generate optimal controls and CBF to reject unsafe trajectories. 

To improve sampling efficiency, we employ the Cross-Entropy Method (CEM) for importance sampling (IS) to sample configurations that will enhance the path with higher probability and store computed optimal gain matrices in a hash table to avoid re-computation during rewiring procedure. We demonstrate the effectiveness of our method on nonlinear control affine systems in simulation.


## Document

The code was tested on Mac OS 13.4 with Python 3.10.0. the required packages are listed in requirements.txt. 

## Simulation and Experiment
### Demonstration of LQR-CBF-RRT*
We designed a challenging navigation scenario where the workspace has six obstacles in close proximity. A robot is required to navigate from the lower left corner to the top right corner without any collisions. Our method is able to generate an optimal and safe trajectory plan efficiently. The result is shown as the following:  
<img width="612" alt="Screenshot 2023-07-25 at 18 30 08" src="https://github.com/gy2256/LQR_CBF_rrtStar/assets/4914320/fe97a0fd-4b88-4b9b-9a5b-c6833e44ff74">


### Hardware Platform
To validate our apparoch, we use a customized DJI Robomaster as our experimental platform. The robot contains four omnidirectional wheels that allow it to move in any direction. The robot takes in velocity command [vx, vy] as the input, and inverse kinematics is calculated to output the corresponding wheel's speed. The robot is equipped with a Raspberry Pi running ROS2 as the communication middleware, and we use an external telemetry tracking system (OptiTrack) for positional tracking to simulate an outdoor GPS-enabled environment. 
![robomaster](https://github.com/gy2256/LQR_CBF_rrtStar/assets/4914320/23220614-9310-456b-9284-09b54919f60a)

### Experiment with real Hardware
We performed an experiment with the real hardware. The robot uses its own online MPC controller to track the generated optimal trajectory. 

https://github.com/gy2256/LQR_CBF_rrtStar/assets/4914320/4c86de33-fe37-4533-afbd-0b2c71a4ca7e



### Citing

If you find this repository useful in your work, please consider citing following work:

```
@article{yang2023efficient,
  title={Efficient LQR-CBF-RRT*: Safe and Optimal Motion Planning},
  author={Yang, Guang and Cai, Mingyu and Ahmad, Ahmad and Belta, Calin and Tron, Roberto},
  journal={arXiv preprint arXiv:2304.00790},
  year={2023}
}
```

## Implementation

cd into linear/nonlinear_dynamic_model
```
python LQR_CBF_rrtStar_..py
```
This allows to identify the performance for both linear and non-linear systems.


## Q/A Contact and Contributors

Guang Yang, Mingyu Cai, Ahmad G. Ahmad
