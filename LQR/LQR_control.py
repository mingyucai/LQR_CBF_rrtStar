import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

np.set_printoptions(precision=3, suppress=True)
max_linear_velocity = 3.0
max_angular_velocity = 1.5708


def getB(yaw, deltat):
    B = np.array([[np.cos(yaw) * deltat, 0], [np.sin(yaw) * deltat, 0], [0, deltat]])
    return B

def state_space_model(A, state_t_minus_1, B, control_input_t_minus_1):
    state_estimate_t = (A @ state_t_minus_1) + (B @ control_input_t_minus_1)
    return state_estimate_t

def lqr(actual_state_x, desired_state_xf, Q, R, A, B, dt):
<<<<<<< HEAD

        x_error = actual_state_x - desired_state_xf

        N = 50

        # Create a list of N + 1 elements
        P = [None] * (N + 1)

        Qf = Q

        # LQR via Dynamic Programming
        P[N] = Qf

        # For i = N, ..., 1
        for i in range(N, 0, -1):
            # Discrete-time Algebraic Riccati equation to calculate the optimal
            # state cost matrix
            P[i - 1] = Q + A.T @ P[i] @ A - (A.T @ P[i] @ B) @ np.linalg.pinv(
                R + B.T @ P[i] @ B) @ (B.T @ P[i] @ A)

            # Create a list of N elements
        K = [None] * N
        u = [None] * N

        # For i = 0, ..., N - 1
        for i in range(N):
            # Calculate the optimal feedback gain K
            K[i] = np.linalg.pinv(R + B.T @ P[i + 1] @ B) @ B.T @ P[i + 1] @ A

            u[i] = K[i] @ np.array(x_error)
            
        # Optimal control input is u_star
        u_star = u[N - 1]

        return u_star
=======
    x_error = actual_state_x - desired_state_xf
    N = 100
    P = [None] * (N + 1)
    Qf = Q
    P[N] = Qf
    for i in range(N, 0, -1):
        P[i-1] = Q + A.T @ P[i] @ A - (A.T @ P[i] @ B) @ np.linalg.pinv(
            R + B.T @ P[i] @ B) @ (B.T @ P[i] @ A)
    K = [None] * N
    u = [None] * N
    for i in range(N):
        K[i] = -np.linalg.pinv(R + B.T @ P[i+1] @ B) @ B.T @ P[i+1] @ A
        u[i] = K[i] @ x_error
    u_star = u[N-1]
    # velocity limitation by scaling the velocity
    if u_star[0] > max_linear_velocity or u_star[1] > max_angular_velocity:
        u_unit = u_star / np.linalg.norm(u_star)
        u_star = u_unit * 5.
    return u_star
>>>>>>> upstream/main



    # return u_star

def main(start, target, verbose=True):
    trajectory = []
    controls = []
    error = []
    waypoints_reached = []

    A = np.array([  [1.0,  0,   0],[  0,1.0,   0],[  0,  0, 1.0]])
    Q = np.array([[1, 0, 0],[0, 1.0, 0], [0, 0, 1.0]])
    R = np.array([[0.01, 0], [0, 0.01]])
    dt = 0.5
    error_th = 0.5

    state_error_magnitude = 1e10
    actual_state_x = start
    desired_state_xf = target
    trajx, trajy = [], []
    while state_error_magnitude >= error_th:

            state_error = actual_state_x - desired_state_xf
            trajectory.append(actual_state_x)

<<<<<<< HEAD
        state_error = actual_state_x - desired_state_xf
        state_error_magnitude = np.linalg.norm(state_error,ord=2)
        print(f'State Error Magnitude = {state_error_magnitude}')
=======
            trajx.append(actual_state_x[0])
            trajy.append(actual_state_x[1])
>>>>>>> upstream/main

            state_error_magnitude = np.linalg.norm(state_error)
            error.append(state_error_magnitude)

            B = getB(actual_state_x[2], dt)
            optimal_control_input = lqr(actual_state_x,
                                        desired_state_xf,
                                        Q, R, A, B, dt)
            controls.append(optimal_control_input)
            if verbose:
                print(f'Control Input = {optimal_control_input}')
            actual_state_x = state_space_model(A, actual_state_x, B,
                                            optimal_control_input)
            if verbose:
                if state_error_magnitude < error_th:
                    print("\nGoal Has Been Reached Successfully!")

    print("trajectory:", trajectory)
    circle = plt.Circle(desired_state_xf[0:2], 0.05)
    ax = plt.gca()
    ax.add_patch(circle)
    ax.plot(trajx, trajy, label="trajectory")
    plt.show()

start = np.array([0,0,0])
target = np.array([5,5,np.pi/2])
main(start, target)

