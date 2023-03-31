import cvxpy
import numpy as np
import matplotlib.pyplot as plt
from math import *
import time
from cvxpy import *
import random

import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from CBFsteer import CBF_RRT
import env, plotting, utils


class MPCplanner:
    def __init__(self):
        self.DT = 0.05  # [s] discrete time
        self.T = 15  # number of horizon
        self.GOAL_DIST = 0.1
        self.MAX_TIME = 100
        # Linear system model
        self.A, self.B = self.get_system_model()

        # initialize CBF
        self.env = env.Env()
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

        self.cbf_rrt_simulation = CBF_RRT(self.obs_circle)

    
    def get_system_model(self):
        A = np.array([[1, 0],[0 , 1]])
        B = np.array([[self.DT, 0], [0, self.DT]])
        return A, B

    def MPC_planning(self, sx, sy, gx, gy, test_MPC = False, cbf_check = True, solve_QP = False):
        self.cbf_rrt_simulation.set_initial_state(np.array([[sx],[sy]]))

        rx, ry = [sx], [sy]
        error = []

        x = np.array([sx, sy])
        goal = np.array([gx, gy])
        u = np.array([0, 0])
        found_path = False

        start_time = time.time()
        Time = 0
        while Time <= self.MAX_TIME:
            Time += self.DT

            ustar, xstar, cost, status = self.MPC_control(x, u, goal)

            if status != OPTIMAL:
                print("Cannot solve MPC opt")
                break

            u[0] = float(ustar[0, 0].value)
            u[1] = float(ustar[1, 0].value)

            # check if LQR control is safe with respect to CBF constraint
            if cbf_check and not test_MPC and not solve_QP:
                if not self.cbf_rrt_simulation.QP_constraint([x[0], x[1]], u, system_type="linear_velocity_control"):
                    break

            x = self.A @ x + self.B @ u

            rx.append(x[0])
            ry.append(x[1])
            error.append(np.linalg.norm(x - goal))

            # check goal
            dis = np.linalg.norm(x - goal)
            if (dis < self.GOAL_DIST):
                # print("Goal")
                found_path = True
                break
        
        print("time of current MPC steering: ", time.time() - start_time)

        if not found_path:
            return rx, ry, error,found_path
        

        return rx, ry, error,found_path



    def MPC_control(self, x0, u, target):
        x = cvxpy.Variable((x0.shape[0], self.T + 1))
        u = cvxpy.Variable((u.shape[0], self.T))
        # MPC controller
        constr = []
        cost = 0
        for t in range(self.T):
            cost += sum_squares(abs(x[0, t] - target[0])) 
            cost += sum_squares(abs(x[1, t] - target[1]))
            constr += [x[:, t + 1] == self.A @ x[:, t] + self.B @ u[:, t], cvxpy.norm(u[:, t], "inf") <= 5]
            # terminal cost
            if t == self.T - 1:
                cost += (x[0, t + 1] - target[0])
                cost += (x[1, t + 1] - target[1]) 
        constr += [x[:, 0] == x0]
        problem = cvxpy.Problem(cvxpy.Minimize(cost), constr)
        problem.solve()
        # print(problem.status)
        return u, x, problem.value, problem.status


if __name__ == '__main__':

    start_time = time.time()
    area = 50.0  # sampling area
    sx = 0.0
    sy = 0.0
    gx = random.uniform(-area, area)
    gy = random.uniform(-area, area)

    mpc_planner = MPCplanner()

    rx, ry, error, foundpath = mpc_planner.MPC_planning(sx, sy, gx, gy, cbf_check=True)

    print(rx[-1], ry[-1])

    print("time of running MPC: ", time.time() - start_time)
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(sx, sy, "or")
    ax1.plot(gx, gy, "ob")
    ax1.plot(rx, ry, "-r")
    ax1.grid()

    ax2.plot(error, label="errors")
    ax2.legend(loc='upper right')
    ax2.grid()

    plt.show()
