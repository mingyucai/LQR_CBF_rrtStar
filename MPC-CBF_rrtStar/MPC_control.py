import cvxpy
import numpy as np
import matplotlib.pyplot as plt
from math import *
import time
from cvxpy import *
import random
        
def main():
    dt = 0.05
    T = 15
    target = [5.0, -1.0] 
    max_speed = 5.0
    min_speed = -5.0


    A = np.array([[1, 0],[0 , 1]])
    B = np.array([[dt, 0], [0, dt]])



    def GetListFromMatrix(x):
        return np.array(x).flatten().tolist()

    def MPC_Opt(A, B, x0, u):
        x_0 = x0[:]
        x = cvxpy.Variable((x0.shape[0], T + 1))
        u = cvxpy.Variable((u.shape[0], T))
        # MPC controller
        states = []
        constr = []
        cost = 0
        for t in range(T):
            cost += sum_squares(abs(x[0, t] - target[0])) 
            cost += sum_squares(abs(x[1, t] - target[1]))
            constr += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t], cvxpy.norm(u[:, t], "inf") <= 5]
            # terminal cost
            if t == T - 1:
                cost += (x[0, t + 1] - target[0])
                cost += (x[1, t + 1] - target[1]) 
        constr += [x[:, 0] == x_0]
        problem = cvxpy.Problem(cvxpy.Minimize(cost), constr)
        result = problem.solve()
        print(problem.status)
        return u, x, problem.value


    x0 = np.array([0.0, 0.0])  # [x,y]
    x = x0
    u = np.array([0.0, 0.0])  # [vx, vy]

    mincost = 100000


    rx, ry = [x[0]], [x[1]]
    error = []

    for i in range(1000):

        ustar, xstar, cost = MPC_Opt(A, B, x, u)
        u[0] = float(ustar[0, 0].value)
        u[1] = float(ustar[1, 0].value)

        x = A @ x + B @ u

        rx.append(x[0])
        ry.append(x[1])
        error.append(np.linalg.norm(x - target))

        # check goal
        dis = np.linalg.norm([x[0] - target[0], x[1] - target[1]])
        if (dis < 0.1):
            print("Goal")
            break

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(x0[0], x0[1], "or")
    ax1.plot(target[0], target[1], "ob")
    ax1.plot(rx, ry, "-r")
    ax1.grid()

    ax2.plot(error, label="errors")
    ax2.legend(loc='upper right')
    ax2.grid()

    plt.show()

if __name__ == '__main__':
    main()