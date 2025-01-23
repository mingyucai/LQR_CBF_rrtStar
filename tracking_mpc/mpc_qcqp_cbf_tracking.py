import numpy as np
import matplotlib.pyplot as plt
import os
import cvxpy
import subprocess
import glob
import gurobipy as gp
from gurobipy import GRB

from Utilits.CubicSpline import cubic_spline_planner, spline_continuity
from Utilits.utils import read_waypoints

"""
Created on Wednesday May 17th, 2023

This code takes into waypoints (positions in 2d) and caclulate the velocity profile between them for MPC tracking


"""


class MPC_controller:
    def __init__(
        self,
        MPC_horizon,
        dt,
        state_weight,
        control_weight,
        x_init,
        obstacles,
        show_animation=True,
        save_animation=False,
    ) -> None:
        self.N = MPC_horizon
        self.dt = dt
        self.Q = state_weight
        self.R = control_weight
        self.obstacles = obstacles
        self.Qf = self.Q
        self.v_lim = 5.0
        self.u_lim = 20.0
        self.x_init = x_init
        self.save_animation = save_animation
        self.show_animation = show_animation
        self.gamma = 5.0  # CBF parameters

        self.state_dimension = 4
        self.control_dimension = 2
        self.goal_dist = 0.2

        self.simulation_max_time = 40.0

    def get_nparray_from_matrix(self, x):
        return np.array(x).flatten()

    def mpc_control(self, x_ref, x_current):
        """
        Input: x_ref: reference trajectory

        Output: State Trajectory and Control Trajectory for the next N steps
        """

        model = gp.Model("MPC_QCQP")
        model.remove(model.getConstrs())
        model.params.NonConvex = 2
        model.Params.LogToConsole = 0
        x = model.addMVar(
            (self.state_dimension, self.N + 1),
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="X",
        )  # [pos_1, vel_1, pos_2, vel_2]
        u = model.addMVar(
            (self.control_dimension, self.N),
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="U",
        )  # [acceleration_1, acceleration_2]

        z = model.addMVar(
            (self.state_dimension, self.N + 1),
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="diff_X",
        )  # [pos_1, vel_1, pos_2, vel_2]

        # System dynamics
        A = np.zeros((self.state_dimension, self.state_dimension))
        B = np.zeros((self.state_dimension, self.control_dimension))

        A[0, 0] = 1.0
        A[0, 1] = self.dt
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[2, 3] = self.dt
        A[3, 3] = 1.0

        B[0, 0] = 0.5 * self.dt**2
        B[1, 0] = self.dt
        B[2, 1] = 0.5 * self.dt**2
        B[3, 1] = self.dt

        # Initial State Constraint
        model.addConstr(x[:, 0] == x_current)

        for t in range(self.N):
            # Dynamics Constraint
            model.addConstr(x[:, t + 1] == A @ x[:, t] + B @ u[:, t])
            model.addConstr(z[:, t] == x_ref[:, t] - x[:, t])

            for obs in self.obstacles:
                model.addConstr(
                    (
                        (x[0, t + 1] - obs[0]) * (x[0, t + 1] - obs[0])
                        + (x[2, t + 1] - obs[1]) * (x[2, t + 1] - obs[1])
                        - obs[2] ** 2
                    )
                    - (
                        (x[0, t] - obs[0]) * (x[0, t] - obs[0])
                        + (x[2, t] - obs[1]) * (x[2, t] - obs[1])
                        - obs[2] ** 2
                    )
                    >= -self.gamma
                    * (
                        (x[0, t] - obs[0]) * (x[0, t] - obs[0])
                        + (x[2, t] - obs[1]) * (x[2, t] - obs[1])
                        - obs[2] ** 2
                    )
                )
            """
            # CBF constraint
            for obs in self.obstacles:
                constraints += [((x[0,t+1]-obs[0])**2 + (x[2,t+1]-obs[1])**2 - obs[2]**2)
                                -((x[0,t]-obs[0])**2 + (x[2,t]-obs[1])**2 - obs[2]**2)
                                >= -self.gamma*((x[0,t]-obs[0])**2 + (x[2,t]-obs[1])**2 - obs[2]**2)]

             """
        # Cost Function
        state_cost = sum((z[:, t]) @ self.Q @ z[:, t] for t in range(self.N + 1))
        control_cost = sum((u[:, t]) @ self.R @ u[:, t] for t in range(self.N))
        model.setObjective(
            state_cost + control_cost + z[:, self.N] @ self.Q @ z[:, self.N],
            GRB.MINIMIZE,
        )
        model.optimize()

        if model.Status == GRB.OPTIMAL:
            solution = model.getVars()
            # X_solution = solution[4*self.N+4:5*self.N+4]
            x1_traj = [item.x for item in solution[: self.N]]
            x2_traj = [item.x for item in solution[self.N + 1 : 2 * self.N + 1]]
            x3_traj = [item.x for item in solution[2 * self.N + 2 : 3 * self.N + 2]]
            x4_traj = [item.x for item in solution[3 * self.N + 3 : 4 * self.N + 3]]
            u1_traj = [item.x for item in solution[4 * self.N + 4 : 5 * self.N + 4]]
            u2_traj = [item.x for item in solution[5 * self.N + 4 : 6 * self.N + 4]]

            return x1_traj, x2_traj, x3_traj, x4_traj, u1_traj, u2_traj
        else:
            print("Cannot find control")
            return [None] * (self.state_dimension + self.control_dimension)

    def intergrate_dynamics_with_euler(self, x_current, u_current):
        # Integrate over a single time step dt using euler method
        # ok to use euler method for linear system
        x_next = x_current

        x_next[0] = (
            x_current[0] + x_current[1] * self.dt + 0.5 * u_current[0] * self.dt**2
        )
        x_next[1] = x_current[1] + u_current[0] * self.dt
        x_next[2] = (
            x_current[2] + x_current[3] * self.dt + 0.5 * u_current[1] * self.dt**2
        )
        x_next[3] = x_current[3] + u_current[1] * self.dt

        return x_next

    def within_goal(self, x_current, x_goal):
        return np.linalg.norm(x_current - x_goal) <= self.goal_dist

    def calculate_local_reference(self, x_ref, t_step):

        # Create an x_local_ref as a shifting window over time.
        x_local_ref = np.zeros((self.state_dimension, self.N + 1))

        if t_step < len(x_ref[0]) - self.N - 1:
            for i in range(self.N + 1):
                x_local_ref[:, i] = x_ref[:, i + t_step]
        else:
            print("running out of reference trajectory")
            for i in range(self.N + 1):
                x_local_ref[:, i] = x_ref[:, -1]
        # Select the closest point to track
        return x_local_ref

    def simulation(self, x_ref):
        time = 0
        x_current = self.x_init.copy()
        simulation_traj = [x_current.copy()]
        u_traj = []

        t_step = 0

        while time < self.simulation_max_time:
            x_local_ref = self.calculate_local_reference(x_ref, t_step)
            x1_traj_N, x2_traj_N, x3_traj_N, x4_traj_N, u1_traj_N, u2_traj_N = (
                self.mpc_control(x_local_ref, x_current)
            )

            if x1_traj_N is None:
                print("Cannot find control")
                break
            else:
                u_current = np.array(
                    [u1_traj_N[0], u2_traj_N[0]]
                )  # Obtain the first control input
                x_current = self.intergrate_dynamics_with_euler(x_current, u_current)
                u_traj.append(u_current.copy())
                simulation_traj.append(x_current.copy())
                time += self.dt

            if self.within_goal([x_current[0], x_current[2]], waypoints[-1]):
                print("Reach goal")
                break

            t_step += 1

            if self.show_animation:
                plt.cla()
                plt.plot(x1_traj_N, x3_traj_N, "-r")
                plt.plot(x_ref[0, :], x_ref[2, :], linestyle="--", color="grey")
                plt.plot(
                    simulation_traj[-1][0],
                    simulation_traj[-1][2],
                    marker="o",
                    color="blue",
                )
                plt.scatter(waypoints[:, 0], waypoints[:, 1], marker="o", color="k")
                plt.title(
                    "Time[s]:"
                    + str(round(time, 2))
                    + ", speed[m/s]:"
                    + str(round(np.sqrt(x_current[1] ** 2 + x_current[3] ** 2), 2))
                )
                plt.axis("equal")
                plt.grid(True)
                plt.pause(0.0001)
                current_directory_path = os.getcwd()
                if self.save_animation:
                    plt.savefig(
                        current_directory_path
                        + "/Animation_images/"
                        + "t_step_"
                        + str(t_step)
                        + ".png"
                    )

        if self.save_animation:
            subprocess.call(
                [
                    "ffmpeg",
                    "-r",
                    "15",
                    "-i",
                    current_directory_path + "/Animation_images" + "/t_step_%01d.png",
                    "-r",
                    "15",
                    "-pix_fmt",
                    "yuv420p",
                    current_directory_path + "/Saved_movies/" + "MPC_tracking.mp4",
                ]
            )

            for file_name in glob.glob(
                current_directory_path + "/Animation_images/" + "*.png"
            ):
                os.remove(file_name)

        return np.array(simulation_traj), np.array(u_traj)

    def calculate_velocity_between_interpolated_positions(
        self, interpolated_x, interpolated_y, target_speed
    ):
        v_interpolated_x = np.zeros(len(interpolated_x))
        v_interpolated_y = np.zeros(len(interpolated_y))

        for i in range(len(interpolated_x) - 1):
            delta_interpolated_x = interpolated_x[i + 1] - interpolated_x[i]
            delta_interpolated_y = interpolated_y[i + 1] - interpolated_y[i]
            angle = np.arctan2(delta_interpolated_y, delta_interpolated_x)
            v_interpolated_x[i] = target_speed * np.cos(angle)
            v_interpolated_y[i] = target_speed * np.sin(angle)

        return v_interpolated_x, v_interpolated_y

    def waypoints_to_x_ref(
        self, waypoints, interpolated_dist, target_speed, interpolation_type="linear"
    ):
        if interpolation_type == "linear":
            rx, ry = [], []
            sp = spline_continuity.Spline2D(
                x=waypoints[:, [0]].flatten(),
                y=waypoints[:, [1]].flatten(),
                kind="linear",
            )
            s = np.arange(0, sp.s[-1], interpolated_dist)
            for i_s in s:
                ix, iy = sp.calc_position(i_s)
                rx.append(ix)
                ry.append(iy)

            interpolated_x1 = np.array(rx)
            interpolated_x3 = np.array(ry)

        elif interpolation_type == "cubic":
            interpolated_x1, interpolated_x3, _, _, _ = (
                cubic_spline_planner.calc_spline_course(
                    x=waypoints[:, [0]].flatten(),
                    y=waypoints[:, [1]].flatten(),
                    ds=interpolated_dist,
                )
            )  # ds is the distance between interpolated points

        x_ref = np.zeros((self.state_dimension, len(interpolated_x1)))
        # Fill reference trajectory with interpolated positions in x1 and x3
        x_ref[0, :] = interpolated_x1[:]
        x_ref[2, :] = interpolated_x3[:]

        # Calculate Velocity between interpolated positions and fill into the x_ref
        v_x2, v_x4 = self.calculate_velocity_between_interpolated_positions(
            interpolated_x1, interpolated_x3, target_speed
        )
        x_ref[1, :] = v_x2[:]
        x_ref[3, :] = v_x4[:]

        # Set the speed at the goal to be zero
        x_ref[1, -1] = 0.0
        x_ref[3, -1] = 0.0

        return x_ref


if __name__ == "__main__":
    plot_border = 32
    target_speed = 10.0  # [m/s]
    interpolated_dist = 0.2  # [m] distance between interpolated position state
    obstacles = [(15.5, 10, 1.0)]  # Circular Obstacles [(x1,x3,radius)]

    path_to_continuous_waypoints = (
        os.getcwd() + "/Saved_continuous_waypoints/state_double_integrator_traj.npy"
    )
    waypoints = read_waypoints(path_to_continuous_waypoints)

    x_init = np.array([waypoints[0, 0], 0.0, waypoints[0, 1], 0.0])  # [p1,v1,p2,v2]

    MPC = MPC_controller(
        MPC_horizon=5,
        dt=0.2,
        state_weight=np.diag([2.0, 0.1, 2.0, 0.1]),  # Q matrix
        control_weight=np.diag([0.1, 0.1]),
        x_init=x_init,
        obstacles=obstacles,
        show_animation=True,
        save_animation=False,
    )  # R matrix

    x_ref = MPC.waypoints_to_x_ref(
        waypoints, interpolated_dist, target_speed, interpolation_type="linear"
    )

    simulation_traj, u_traj = MPC.simulation(x_ref)

    fig, ax = plt.subplots()
    ax.plot(
        simulation_traj[:, [0]],
        simulation_traj[:, [2]],
        marker="o",
        linestyle="-",
        color="blue",
        label="MPC Path",
    )
    ax.plot(
        x_ref[0].flatten(),
        x_ref[2].flatten(),
        marker=".",
        linestyle="--",
        color="grey",
        label="x_ref",
    )
    ax.scatter(
        waypoints[:, [0]],
        waypoints[:, [1]],
        marker="o",
        color="black",
        label="waypoints",
        s=100,
    )
    ax.set_xticks(range(plot_border))
    ax.set_yticks(range(plot_border))
    ax.set_xlim([-0.5, plot_border - 0.5])
    ax.set_ylim([-0.5, plot_border - 0.5])

    ax.set_title("MPC Tracking")
    ax.set_xlabel("X1-coordinate")
    ax.set_ylabel("X3-coordinate")
    ax.legend()
    plt.show()
