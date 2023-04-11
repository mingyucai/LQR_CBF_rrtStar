## Introduction

Control Barrier Functions (CBF) are a powerful tool for designing safety-critical controllers and motion planners. The safety requirements are encoded as a continuously differentiable function that maps from state variables to a real value, in which the sign of its output determines whether safety is violated. In practice, the CBFs can be used to enforce safety by imposing itself as a constraint in a Quadratic Program (QP) solved point-wise in time. However, this approach costs computational resources and could lead to infeasibility in solving the QP. 

In this project, we propose a novel motion planning framework that combines sampling-based methods with Linear Quadratic Regulator (LQR) and CBFs. Our approach does not require solving the QPs for control synthesis and avoids explicit collision checking during samplings. Instead, it uses LQR to generate optimal controls and CBF to reject unsafe trajectories. 

To improve sampling efficiency, we employ the Cross-Entropy Method (CEM) for importance sampling (IS) to sample configurations that will enhance the path with higher probability and store computed optimal gain matrices in a hash table to avoid re-computation during rewiring procedure. We demonstrate the effectiveness of our method on nonlinear control affine systems in simulation.

## Document

Under construction


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
This allows to identify the performance for both linear and nonlineary systems


## Q/A Contact and Contributors

Mingyu Cai, Guang Yang, Ahmad G. Ahmad