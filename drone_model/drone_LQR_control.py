import numpy as np
from scipy.integrate import odeint
import control
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class DroneParams:
    mass: float = 0.028
    gravity: float = 9.81
    I_xx: float = 16.571710e-6
    I_yy: float = 16.655602e-6
    I_zz: float = 29.261652e-6


class LQRController:
    def __init__(self):
        self.drone = DroneParams()
        self.N = 12  # state dimension
        self.M = 4  # control dimension

        #Linearized state space model 
        self.A = np.array(
            [
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # phi
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # theta
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # psi
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # p
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # q
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # r
                [0, self.drone.gravity, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x_dot
                [-self.drone.gravity, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # y_dot
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # z_dot
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # x
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # y
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # z
            ]
        )

        self.B = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 1 / self.drone.I_xx, 0, 0],
                [0, 0, 1 / self.drone.I_yy, 0],
                [0, 0, 0, 1 / self.drone.I_zz],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1 / self.drone.mass, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )

        # LQR gains
        self.Q = np.diag([100, 100, 100, 10, 10, 10, 10, 10, 10, 1, 1, 1])
        self.R = np.diag([0.1, 0.1, 0.1, 0.1])
        self.K, _, _ = control.lqr(self.A, self.B, self.Q, self.R)

    def get_control(self, current_state, goal_state):
        x_error = current_state.reshape(self.N, 1) - goal_state.reshape(self.N, 1)
        ud = np.array([[self.drone.mass * self.drone.gravity, 0, 0, 0]]).T
        u = -self.K @ x_error + ud
        return u.flatten()