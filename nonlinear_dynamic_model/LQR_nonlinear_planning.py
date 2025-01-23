import math
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy
import time

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from CBFsteer import CBF_RRT
import env, plotting, utils

SHOW_ANIMATION = False


class LQRPlanner:

    def __init__(self):

        self.N = 3  # number of state variables
        self.M = 2  # number of control variables
        self.DT = 0.2  # discretization step

        self.MAX_TIME = 100.0  # Maximum simulation time
        self.GOAL_DIST = 0.6
        self.MAX_ITER = 250
        self.EPS = 0.01

        # initialize CBF
        self.env = env.Env()
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

        self.cbf_rrt_simulation = CBF_RRT(self.obs_circle)

    def lqr_planning(
        self,
        sx,
        sy,
        gx,
        gy,
        LQR_gain,
        test_LQR=False,
        show_animation=True,
        cbf_check=True,
        solve_QP=False,
    ):

        # Linearize system model
        xd = np.matrix([[gx], [gy], [np.pi / 4]])
        ud = np.matrix([[0], [0]])
        self.A, self.B, self.C = self.get_linear_model(xd, ud)
        # LQR gain is invariant
        Q = np.matrix("0.5 0 0; 0 1 0; 0 0 0.01")
        R = np.matrix("0.01 0; 0 0.01")

        # check the hash table to store LQR feedback Gain
        waypoint = (gx, gy)
        if waypoint in LQR_gain:
            # print('found one prebious gain')
            self.K = LQR_gain[waypoint]
        else:
            self.K = self.dLQR(self.A, self.B, Q, R)
            LQR_gain[waypoint] = self.K

        self.cbf_rrt_simulation.set_initial_state(np.array([[sx], [sy]]))

        rx, ry = [sx], [sy]

        stheta, gtheta = 0, np.pi / 4

        error = []

        # x = np.array([sx - gx, sy - gy, stheta - gtheta]).reshape(3, 1)  # State vector

        xk = np.array([sx, sy, stheta]).reshape(3, 1)  # State vector

        found_path = False

        time = 0.0
        while time <= self.MAX_TIME:
            time += self.DT

            x = xk - xd
            u = self.K @ x

            # check if LQR control is safe with respect to CBF constraint
            if cbf_check and not test_LQR and not solve_QP:
                if not self.cbf_rrt_simulation.QP_constraint(
                    [x[0, 0] + gx, x[1, 0] + gy, x[2, 0] + gtheta],
                    u,
                    system_type="unicycle_velocity_control",
                ):
                    break

            if solve_QP:
                try:
                    u = self.cbf_rrt_simulation.QP_controller(
                        [x[0, 0] + gx, x[1, 0] + gy, x[2, 0] + gtheta],
                        u,
                        model="unicycle",
                    )
                except:
                    print("The CBF-QP at current steering step is infeasible")
                    break

            xk = self.A @ xk + self.B @ u + self.C

            rx.append(xk[0, 0])
            ry.append(xk[1, 0])

            d = math.sqrt((gx - rx[-1]) ** 2 + (gy - ry[-1]) ** 2)
            error.append(d)

            if d <= self.GOAL_DIST:
                found_path = True
                # print('errors ', d)
                break

            # animation
            if show_animation:  # pragma: no cover
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect(
                    "key_release_event",
                    lambda event: [exit(0) if event.key == "escape" else None],
                )
                plt.plot(sx, sy, "or")
                plt.plot(gx, gy, "ob")
                plt.plot(rx, ry, "-r")
                plt.axis("equal")
                plt.pause(1.0)

        if not found_path:
            # print("Cannot found path")
            return rx, ry, error, found_path

        return rx, ry, error, found_path

    def dLQR(self, A, B, Q, R):

        N = 50

        # Create a list of N + 1 elements
        P = [None] * (N + 1)

        Qf = Q

        # LQR via Dynamic Programming
        P[N] = Qf

        # For i = N, ..., 1
        for i in range(N, 0, -1):
            # state cost matrix
            P[i - 1] = (
                Q
                + A.T @ P[i] @ A
                - (A.T @ P[i] @ B)
                @ np.linalg.pinv(R + B.T @ P[i] @ B)
                @ (B.T @ P[i] @ A)
            )

        # Create a list of N elements
        K = [None] * N
        u = [None] * N

        P1 = P[N - 1]

        K1 = -np.linalg.inv(R + B.T @ P[N] @ B) @ B.T @ P[N] @ A

        return K1

    def get_linear_model(self, x_bar, u_bar):
        """
        Computes the LTI approximated state space model x' = Ax + Bu + C
        """

        x = x_bar[0]
        y = x_bar[1]
        theta = x_bar[2]

        v = u_bar[0]
        yaw = u_bar[1]

        A = np.zeros((self.N, self.N))
        A[0, 2] = -v * np.sin(theta)
        A[1, 2] = v * np.sin(theta)
        A_lin = np.eye(self.N) + self.DT * A

        B = np.zeros((self.N, self.M))
        B[0, 0] = np.cos(theta)
        B[1, 0] = np.sin(theta)
        B[2, 1] = 1
        B_lin = self.DT * B

        f_xu = np.array([v * np.cos(theta), v * np.sin(theta), theta]).reshape(
            self.N, 1
        )
        C_lin = self.DT * (
            f_xu
            - np.dot(A, x_bar.reshape(self.N, 1))
            - np.dot(B, u_bar.reshape(self.M, 1))
        )

        return np.round(A_lin, 4), np.round(B_lin, 4), np.round(C_lin, 4)


class LQRPlanner_acceleration:

    def __init__(self):
        self.N = 4  # number of state variables
        self.M = 2  # number of control variables
        self.DT = 0.2  # discretization step

        self.MAX_TIME = 250.0  # Maximum simulation time
        self.GOAL_DIST = 0.5
        self.MAX_ITER = 250
        self.EPS = 0.01

        # initialize CBF
        self.env = env.Env()
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

        self.cbf_rrt_simulation = CBF_RRT(self.obs_circle)

    def lqr_planning(
        self,
        sx,
        sy,
        sv,
        stheta,
        gx,
        gy,
        gv,
        gtheta,
        test_LQR=False,
        show_animation=True,
        cbf_check=True,
    ):

        # Linearize system model
        gv, gtheta = 0.1, np.pi / 4
        xd = np.matrix([[gx], [gy], [gv], [np.pi / 4]])
        ud = np.matrix([[0], [0]])
        self.A, self.B, self.C = self.get_linear_acc_model(xd, ud)
        # LQR gain is invariant
        Q = np.matrix("1 0 0 0; 0 1 0 0; 0 0 0.05 0; 0 0 0 0.01")
        R = np.matrix("0.01 0; 0 0.01")
        self.K = self.dLQR(self.A, self.B, Q, R)

        self.cbf_rrt_simulation.set_initial_state(np.array([[sx], [sy]]))

        rx, ry, rv, rtheta = [sx], [sy], [sv], [stheta]

        error = []

        # x = np.array([sx - gx, sy - gy, sv - gv, stheta - gtheta]).reshape(4, 1)  # State vector
        xk = np.array([sx, sy, sv, stheta]).reshape(4, 1)  # State vector

        found_path = False

        time = 0.0
        while time <= self.MAX_TIME:
            time += self.DT

            x = xk - xd

            u = self.K @ x

            # check if LQR control is safe with respect to CBF constraint
            if cbf_check and not test_LQR:
                if not self.cbf_rrt_simulation.QP_constraint(
                    [x[0, 0] + gx, x[1, 0] + gy, x[2, 0] + gv, x[3, 0] + gtheta],
                    u,
                    system_type="unicycle_acceleration_control",
                ):
                    # print("violation")
                    break

            xk = self.A @ xk + self.B @ u + self.C

            rx.append(xk[0, 0])
            ry.append(xk[1, 0])

            d = math.sqrt((gx - rx[-1]) ** 2 + (gy - ry[-1]) ** 2)
            error.append(d)

            if d <= self.GOAL_DIST:
                found_path = True
                # print('errors ', d)
                break

            # animation
            if show_animation:  # pragma: no cover
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect(
                    "key_release_event",
                    lambda event: [exit(0) if event.key == "escape" else None],
                )
                plt.plot(sx, sy, "or")
                plt.plot(gx, gy, "ob")
                plt.plot(rx, ry, "-r")
                plt.axis("equal")
                plt.pause(1.0)

        if not found_path:
            # print("Cannot found path")
            return rx, ry, error, found_path

        return rx, ry, error, found_path

    def dLQR(self, A, B, Q, R):

        N = 50

        # Create a list of N + 1 elements
        P = [None] * (N + 1)

        Qf = Q

        # LQR via Dynamic Programming
        P[N] = Qf

        # For i = N, ..., 1
        for i in range(N, 0, -1):

            # state cost matrix
            P[i - 1] = (
                Q
                + A.T @ P[i] @ A
                - (A.T @ P[i] @ B)
                @ np.linalg.pinv(R + B.T @ P[i] @ B)
                @ (B.T @ P[i] @ A)
            )

        # Create a list of N elements
        K = [None] * N
        u = [None] * N

        P1 = P[N - 1]

        K1 = -np.linalg.inv(R + B.T @ P[N] @ B) @ B.T @ P[N] @ A

        return K1

    # nonlinear acceleration model
    def get_linear_acc_model(self, x_bar, u_bar):
        """
        Computes the LTI approximated state space model x' = Ax + Bu + C
        """

        L = 0.3  # vehicle wheelbase

        x = x_bar[0]
        y = x_bar[1]
        v = x_bar[2]
        theta = x_bar[3]

        a = u_bar[0]
        delta = u_bar[1]

        A = np.zeros((self.N, self.N))
        A[0, 2] = np.cos(theta)
        A[0, 3] = -v * np.sin(theta)
        A[1, 2] = np.sin(theta)
        A[1, 3] = v * np.cos(theta)
        A[3, 2] = 0
        A_lin = np.eye(self.N) + self.DT * A

        B = np.zeros((self.N, self.M))
        B[2, 0] = 1
        B[3, 1] = 1
        B_lin = self.DT * B

        f_xu = np.array(
            [v * np.cos(theta), v * np.sin(theta), a, v * np.tan(delta) / L]
        ).reshape(self.N, 1)
        C_lin = self.DT * (
            f_xu
            - np.dot(A, x_bar.reshape(self.N, 1))
            - np.dot(B, u_bar.reshape(self.M, 1))
        )

        return np.round(A_lin, 4), np.round(B_lin, 4), np.round(C_lin, 4)


def main():
    print(__file__ + " start!!")

    ntest = 1  # number of goal
    area = 50.0  # sampling area

    acceleration_model = True

    gx = 15.0  # random.uniform(-area, area)
    gy = 15.0  # random.uniform(-area, area)

    if not acceleration_model:
        lqr_planner = LQRPlanner()
    else:
        lqr_planner = LQRPlanner_acceleration()

    for i in range(ntest):
        start_time = time.time()
        sx = 0.0
        sy = 0.0

        print("goal", gy, gx)

        if not acceleration_model:
            rx, ry, error, foundpath = lqr_planner.lqr_planning(
                sx, sy, gx, gy, test_LQR=True, show_animation=SHOW_ANIMATION
            )

        else:
            sv, stheta = 0, 0
            gv, gtheta = 0.1, np.pi / 4
            rx, ry, error, foundpath = lqr_planner.lqr_planning(
                sx,
                sy,
                sv,
                stheta,
                gx,
                gy,
                gv,
                gtheta,
                test_LQR=True,
                show_animation=SHOW_ANIMATION,
            )

        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

        print("time of running LQR: ", time.time() - start_time)

        ax1.plot(sx, sy, "or")
        ax1.plot(gx, gy, "ob")
        ax1.plot(rx, ry, "-r")
        ax1.grid()

        ax2.plot(error, label="errors")
        ax2.legend(loc="upper right")
        ax2.grid()
        plt.show()

        if SHOW_ANIMATION:  # pragma: no cover
            plt.plot(sx, sy, "or")
            plt.plot(gx, gy, "ob")
            plt.plot(rx, ry, "-r")
            plt.axis("equal")
            plt.pause(1.0)


if __name__ == "__main__":

    main()
