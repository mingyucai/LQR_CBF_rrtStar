import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


np.set_printoptions(precision=3, suppress=True)

# Optional Variables
max_linear_velocity = 3.0
max_angular_velocity = 3.0


def getB(deltat):

    B = np.array([[deltat, 0],
                  [deltat, 0]])
    return B

def state_space_model(A, state_t_minus_1, B, control_input_t_minus_1):

    control_input_t_minus_1[0] = np.clip(control_input_t_minus_1[0],
                                         -max_linear_velocity,
                                         max_linear_velocity)
    control_input_t_minus_1[1] = np.clip(control_input_t_minus_1[1],
                                         -max_angular_velocity,
                                         max_angular_velocity)
    state_estimate_t = (A @ state_t_minus_1) + (B @ control_input_t_minus_1)

    return state_estimate_t

def lqr(actual_state_x, desired_state_xf, Q, R, A, B, dt):

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
            K[i] = -np.linalg.pinv(R + B.T @ P[i + 1] @ B) @ B.T @ P[i + 1] @ A

            u[i] = K[i] @ x_error

        # Optimal control input is u_star
        u_star = u[N - 1]

        return u_star


def main():
    # Let the time interval be 1.0 seconds
    dt = 0.05

    actual_state_x = np.array([0, 0])

    # Desired state [x,y,yaw angle]
    # [meters, meters, radians]
    desired_state_xf = np.array([10.000, 10.000])

    A = np.array([[1.0, 0.],
                  [0., 1.0]])

    R = np.array([[0.01, 0],  # Penalty for linear velocity effort
                  [0, 0.01]])  # Penalty for angular velocity effort

    Q = np.array([[1.0, 0],  # Penalize X position error
                  [0, 1.0]])

    # Launch the robot, and have it move to the desired goal destination
    trajx, trajy = [], []
    for i in range(500):
        print(f'iteration = {i} seconds')
        print(f'Current State = {actual_state_x}')
        print(f'Desired State = {desired_state_xf}')

        trajx.append(actual_state_x[0])
        trajy.append(actual_state_x[1])

        state_error = actual_state_x - desired_state_xf
        state_error_magnitude = np.linalg.norm(state_error)
        print(f'State Error Magnitude = {state_error_magnitude}')

        B = getB(dt)

        # LQR returns the optimal control input
        optimal_control_input = lqr(actual_state_x,
                                    desired_state_xf,
                                    Q, R, A, B, dt)

        print(f'Control Input = {optimal_control_input}')

        # We apply the optimal control to the robot
        # so we can get a new actual (estimated) state.
        actual_state_x = state_space_model(A, actual_state_x, B,
                                           optimal_control_input)

        # Stop as soon as we reach the goal
        # Feel free to change this threshold value.
        if state_error_magnitude < 0.01:
            print("\nGoal Has Been Reached Successfully!")
            break

        print()

    circle = plt.Circle(desired_state_xf[0:2], 0.05)
    ax = plt.gca()
    ax.add_patch(circle)
    ax.plot(trajx, trajy, label="trajectory")
    plt.show()

main()
