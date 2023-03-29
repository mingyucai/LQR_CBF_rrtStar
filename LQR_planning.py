import math
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy
import time

from CBFsteer import CBF_RRT
import env, plotting, utils

SHOW_ANIMATION = False


class LQRPlanner:

    def __init__(self):
        self.MAX_TIME = 100.0  # Maximum simulation time
        self.DT = 0.05  # Time tick
        self.GOAL_DIST = 0.1
        self.MAX_ITER = 150
        self.EPS = 0.01

         # Linear system model
        self.A, self.B = self.get_system_model()
        # LQR gain is invariant
        self.K = self.lqr_control(self.A, self.B)

        # initialize CBF
        self.env = env.Env()
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

        self.cbf_rrt_simulation = CBF_RRT(self.obs_circle)

    def lqr_planning(self, sx, sy, gx, gy, test_LQR = False, show_animation = True, cbf_check = True, solve_QP = True):

        self.cbf_rrt_simulation.set_initial_state(np.array([[sx],[sy]]))

        rx, ry = [sx], [sy]

        error = []

        x = np.array([sx - gx, sy - gy]).reshape(2, 1)  # State vector

        found_path = False


        time = 0.0
        while time <= self.MAX_TIME:
            time += self.DT

            u = self.K @ x

            # check if LQR control is safe with respect to CBF constraint
            if cbf_check and not test_LQR and not solve_QP:
                if not self.cbf_rrt_simulation.QP_constraint([x[0, 0] + gx, x[1, 0] + gy],u):
                    break
            
            if solve_QP:
                try:
                    u = self.cbf_rrt_simulation.QP_controller(x, u, model="linear")
                except:
                    break

            x = self.A @ x + self.B @ u

            rx.append(x[0, 0] + gx)
            ry.append(x[1, 0] + gy)


            d = math.sqrt((gx - rx[-1]) ** 2 + (gy - ry[-1]) ** 2)
            error.append(d)

            if d <= self.GOAL_DIST:
                found_path = True
                # print('errors ', d)
                break

            # animation
            if show_animation:  # pragma: no cover
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                        lambda event: [exit(0) if event.key == 'escape' else None])
                plt.plot(sx, sy, "or")
                plt.plot(gx, gy, "ob")
                plt.plot(rx, ry, "-r")
                plt.axis("equal")
                plt.pause(1.0)

        if not found_path:
            #print("Cannot found path")
            return rx, ry, error,found_path

        return rx, ry, error,found_path


    def dlqr(self, A,B,Q,R):
        """
        Solve the discrete time lqr controller.
        x[k+1] = A x[k] + B u[k]
        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
        """
        # first, solve the ricatti equation
        P = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
        # compute the LQR gain
        K = np.matrix(scipy.linalg.inv(B.T*P*B+R)*(B.T*P*A))

        eigVals, eigVecs = scipy.linalg.eig(A-B*K)
        
        return -K

    def get_system_model(self):

        A = np.matrix([[1, 0],[0 , 1]])

        B = np.matrix([[self.DT, 0], [0, self.DT]])


        return A, B

    def lqr_control(self, A, B):

        Q = np.matrix("1 0; 0 1")
        R = np.matrix("0.01 0; 0 0.01")

        Kopt = self.dlqr(A, B, Q, R)


        return Kopt
    


class LQRPlanner_acceleration:

    def __init__(self):
        self.MAX_TIME = 100.0  # Maximum simulation time
        self.DT = 0.05  # Time tick
        self.GOAL_DIST = 0.1
        self.MAX_ITER = 150
        self.EPS = 0.01

         # Linear system model
        self.A, self.B = self.get_system_model()
        # LQR gain is invariant
        self.K = self.lqr_control(self.A, self.B)

        # initialize CBF
        self.env = env.Env()
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

        self.cbf_rrt_simulation = CBF_RRT(self.obs_circle)

    def lqr_planning(self, sx, sy, svx, svy, gx, gy, gvx, gvy, test_LQR = False, show_animation = True, cbf_check = True):

        self.cbf_rrt_simulation.set_initial_state(np.array([[sx],[sy]]))

        rx, ry, rvx, rvy = [sx], [sy], [svx], [svy]

        error = []

        x = np.array([sx - gx, sy - gy, svx - gvx, svy - gvy]).reshape(4, 1)  # State vector

        found_path = False


        time = 0.0
        while time <= self.MAX_TIME:
            time += self.DT

            u = self.K @ x

            # check if LQR control is safe with respect to CBF constraint
            if cbf_check and not test_LQR:

                if not self.cbf_rrt_simulation.QP_constraint([x[0, 0] + gx, x[1, 0] + gy, x[2, 0] + gvx, x[3, 0] + gvy], u, system_type = "linear_acceleration_control"):
                    # print("violation")
                    # exit(0)
                    break


            x = self.A @ x + self.B @ u

            rx.append(x[0, 0] + gx)
            ry.append(x[1, 0] + gy)


            d = math.sqrt((gx - rx[-1]) ** 2 + (gy - ry[-1]) ** 2)
            error.append(d)

            if d <= self.GOAL_DIST:
                found_path = True
                # print('errors ', d)
                break

            # animation
            if show_animation:  # pragma: no cover
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                        lambda event: [exit(0) if event.key == 'escape' else None])
                plt.plot(sx, sy, "or")
                plt.plot(gx, gy, "ob")
                plt.plot(rx, ry, "-r")
                plt.axis("equal")
                plt.pause(1.0)

        if not found_path:
            #print("Cannot found path")
            return rx, ry, error,found_path

        return rx, ry, error,found_path


    def dlqr(self, A,B,Q,R):
        """
        Solve the discrete time lqr controller.
        x[k+1] = A x[k] + B u[k]
        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
        """
        # first, solve the ricatti equation
        P = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
        # compute the LQR gain
        K = np.matrix(scipy.linalg.inv(B.T*P*B+R)*(B.T*P*A))

        eigVals, eigVecs = scipy.linalg.eig(A-B*K)
        
        return -K

    def get_system_model(self):

        A = np.matrix([[1, 0, self.DT, 0],[0 , 1, 0, self.DT], [0, 0, 1, 0], [0, 0, 0, 1]])

        B = np.matrix([[0, 0], [0, 0], [self.DT, 0], [0, self.DT]])

        return A, B

    def lqr_control(self, A, B):

        Q = np.matrix("1 0 0 0; 0 1 0 0; 0 0 0.1 0; 0 0 0 0.1")
        R = np.matrix("0.01 0; 0 0.01")

        Kopt = self.dlqr(A, B, Q, R)


        return Kopt
    


def main():
    print(__file__ + " start!!")

    ntest = 1  # number of goal
    area = 50.0  # sampling area

    acceleration_model = True

    if not acceleration_model:
        lqr_planner = LQRPlanner()
    else:
        lqr_planner = LQRPlanner_acceleration()

    for i in range(ntest):
        start_time = time.time()
        sx = 6.0
        sy = 6.0
        gx = random.uniform(-area, area)
        gy = random.uniform(-area, area)
        print("goal", gy, gx)

        if not acceleration_model:
            rx, ry, error, foundpath = lqr_planner.lqr_planning(sx, sy, gx, gy, test_LQR = True, show_animation=SHOW_ANIMATION)
        
        else:
            svx, svy, gvx, gvy = 0, 0, 0, 0
            rx, ry, error, foundpath = lqr_planner.lqr_planning(sx, sy, svx, svy, gx, gy, gvx, gvy, test_LQR = True, show_animation=SHOW_ANIMATION)


        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

        print("time of running LQR: ", time.time() - start_time)

        ax1.plot(sx, sy, "or")
        ax1.plot(gx, gy, "ob")
        ax1.plot(rx, ry, "-r")
        ax1.grid()
        
        ax2.plot(error, label="errors")
        ax2.legend(loc='upper right')
        ax2.grid()
        plt.show()


        if SHOW_ANIMATION:  # pragma: no cover
            plt.plot(sx, sy, "or")
            plt.plot(gx, gy, "ob")
            plt.plot(rx, ry, "-r")
            plt.axis("equal")
            plt.pause(1.0)


if __name__ == '__main__':

    main()