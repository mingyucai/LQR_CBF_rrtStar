
import matplotlib.pyplot as plt
import math
import numpy as np
import heapq as hq
import sys
import sympy

from numpy import linalg as LA
from scipy.integrate import solve_ivp
from gurobipy import *

"""
Created on Mon Oct 24 9:18 PM
@author: Guang Yang
This code is a rewritten version of the original CBF-RRT paper with
linear dynamics
"""
class Obstacle_Sphere(object):
    def __init__(self, center, radius):
        self.T = 1.0  #Integration Length
        self.N = 50 # Number of Control Updates
        self.center=center
        self.radius=radius
        

    def h(self,x):
        return LA.norm(x-self.center,2)**2-self.radius**2

    def gradh(self,x):
        return 2*(x-self.center)

    def hdot(self,x,xdot):
        return self.gradh(x).transpose().dot(xdot)

def fun_derivative_trajectory(x,dx,f,gradf):
    fx=np.apply_along_axis(f,0,x)
    gradfx=np.apply_along_axis(gradf,0,x)
    dfx=np.sum(gradfx*dx,0)

    #plot fx and dfx
    t_span = np.linspace(0,7.0,100)

    fig, ax = plt.subplots()
    ax.plot(t_span, -fx,'r',label="-h")
    #ax.plot(t_span, gradfx[0,:],'g')
    ax.plot(t_span, dfx,'b',label="h_dot")
    ax.set_xlabel("Time")
    ax.legend()

    plt.show()

class CBF_RRT:
    def __init__(self, obstacle_list):
        self.t0 = 0 # Starting time
        self.T = 0.2  #Integration Length
        self.N = 50 # Number of Control Updates
        # self.y0 = initial_state
        self.k = 6 # k nearest neighbor obstacles that will be used for generating CBF constraint
        self.cbf_constraints_sensing_radius = 20
        self.k_cbf = 1.0 #CBF coefficient for double intergrators
        self.k_cbf2 = 2.0 #CBF coefficient for double intergrators, If acceleration is used
        self.p_cbf = 1 #CBF constraint power
        self.k1_unicyle_cbf = 2.0 # CBF coefficient for unicycle
        self.k2_unicyle_cbf = 2.0 # CBF coefficient for unicycle

        self.x_obstacle = obstacle_list
        self.u1_lower_lim = -5
        self.u1_upper_lim = 5
        self.u2_lower_lim = -5
        self.u2_upper_lim = 5
        self.w_lower_lim = -5
        self.w_upper_lim = 5
        self.unicycle_constant_v = 1.0
        self.unicycle_gamma = 0.5

        self.u1_traj = np.zeros(shape=(0,0))
        self.u2_traj = np.zeros(shape=(0,0))
        self.x1_traj = np.zeros(shape=(0,0))
        self.x2_traj = np.zeros(shape=(0,0))

        self.cbf_traj = np.zeros(shape=(0,0))
        self.hdot_traj = np.zeros(shape=(0,0))
        self.h_traj = np.zeros(shape=(0,0))
    
    def set_initial_state(self, initial_state):
        self.y0 = initial_state

    def CLF_unicycle_QP(self,x_current, x_goal):
        # CLF QP for unicycle model with angular velocity as control
        x = x_current[0]
        y = x_current[1]
        theta = x_current[2]

        xd = x_goal[0]
        yd = x_goal[1]


        self.m = Model("Unicycle_CLF_QP")
        self.m.remove(self.m.getConstrs())

        V = (math.cos(theta)*(y-yd)-math.sin(theta)*(x-xd))**2
        LgV = -2*(math.cos(theta)*(x-xd)+math.sin(theta)*(y-yd))*(math.cos(theta)*(y-yd)-math.sin(theta)*(x-xd))

        self.w = self.m.addVar(lb=self.w_lower_lim, ub=self.w_upper_lim,
            vtype=GRB.CONTINUOUS, name="Angular velocity Constraint")
        
        # Slack variable
        self.delta = self.m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS, name="Slack Variable")
        
        # Initialize Cost Function
        self.cost_func = self.w*self.w+self.delta*self.delta
        self.m.setObjective(self.cost_func, GRB.MINIMIZE)

        self.m.addConstr((-LgV*self.w-self.unicycle_gamma*V+self.delta)>=0)

         #Stop optimizer from publsihing results to console - remove if desired
        self.m.Params.LogToConsole = 0

        #Solve the optimization problem
        self.m.optimize()
        solution = self.m.getVars()
        u_ref = solution[0].x

        return u_ref
        

    def QP_controller(self, x_current, u_ref, model="linear"):
        if model == "linear":
            self.m = Model("CBF_CLF_QP_Linear")
            x1 = x_current[0, 0]
            x2 = x_current[1, 0]

            self.m.remove(self.m.getConstrs())
            u1_ref = u_ref[0]
            u2_ref = u_ref[1]
            # Control Acutuator Constraints
            self.u1 = self.m.addVar(lb=self.u1_lower_lim, ub=self.u1_upper_lim,
            vtype=GRB.CONTINUOUS, name="velocity_constraint_x1")
            self.u2 = self.m.addVar(lb=self.u2_lower_lim, ub=self.u2_upper_lim,
            vtype=GRB.CONTINUOUS, name="velocity_constraint_x2")

            # Initialize Cost Function
            self.cost_func = (self.u1-u1_ref)*(self.u1-u1_ref)+(self.u2-u2_ref)*(self.u2-u2_ref)
            self.m.setObjective(self.cost_func, GRB.MINIMIZE)

            # CBF function
            for i in range(0,len(self.x_obstacle)):
                h = (x1-self.x_obstacle[i][0])**2+(x2-self.x_obstacle[i][1])**2-self.x_obstacle[i][2]**2

                lgh = 2*(x1-self.x_obstacle[i][0])*self.u1+2*(x2-self.x_obstacle[i][1])*self.u2

                self.m.addConstr((lgh+self.k_cbf*h**self.p_cbf)>=0)


            #Stop optimizer from publsihing results to console - remove if desired
            self.m.Params.LogToConsole = 0

            #Solve the optimization problem
            self.m.optimize()
            self.solution = self.m.getVars()
            #self.m.write("Safe_RRT_Forward.lp")

            # get the results of decision variables
            u1 = self.solution[0].x
            u2 = self.solution[1].x
            return np.array([[u1],[u2]])
        
        if model == "unicycle":
            # TO DO : Make sure if we want Lyapunov function
            self.m = Model("CBF_CLF_QP_Unicycle")
            x = x_current[0]
            y = x_current[1]
            theta = x_current[2]


            v = u_ref[0, 0]  #self.unicycle_constant_v # Set constant linear velcoity to avoid mixed relative degree control
            yaw = u_ref[1, 0]

            self.m.remove(self.m.getConstrs())
            #Control angular velocity
            self.w = self.m.addVar(lb=self.w_lower_lim, ub=self.w_upper_lim,vtype=GRB.CONTINUOUS, name="Control_Angular_Velocity")

            # Initialize Cost Function, minimize distanc to u_ref
            self.cost_func = (self.w-yaw)*(self.w-yaw)
            self.m.setObjective(self.cost_func, GRB.MINIMIZE)

            # CBF Constraint for h(x) = (x1 + x_{obs,1})^2 + (x2 + x_{obs,2})^2 - r^2>= 0
            for i in range(0,len(self.x_obstacle)):
                xo = self.x_obstacle[i][0]
                yo = self.x_obstacle[i][1]
                r = self.x_obstacle[i][2]

                h = (x-xo)**2+(y-yo)**2-r**2

                Lfh = v*math.cos(theta)*(2*x-2*xo)+v*math.sin(theta)*(2*y-2*yo)

                LfLfh = 2*(v**2)*math.cos(theta)**2+2*(v**2)*math.sin(theta)**2

                LgLfh = v*math.cos(theta)*(2*y-2*yo) - v*math.sin(theta)*(2*x-2*xo)

                self.m.addConstr( LfLfh+LgLfh*self.w+self.k1_unicyle_cbf*h+self.k2_unicyle_cbf*Lfh >= 0,"CBF_constraint")

            #Stop optimizer from publsihing results to console - remove if desired
            self.m.Params.LogToConsole = 0
            
            #Solve the optimization problem
            self.m.optimize()
            self.solution = self.m.getVars()
            u = self.solution[0].x

            return np.array([v, u]).reshape(2,1)

    def find_knn_obstacle(self, x_current, x_obstacles,k):
        '''
            Take in current node's position and obstacles list and integer k
            return k obstacles index based on distance (from close to far)
        '''

        k_nn_x_obstacles_idx = []
        obstacle_distance_pq = []
        hq.heapify(obstacle_distance_pq)

        for i in range(len(x_obstacles)):
            distance = math.hypot(x_current[0] - x_obstacles[i][0], x_current[1] - x_obstacles[i][1])
            hq.heappush(obstacle_distance_pq, (distance,i))
        
        for i in range(0,k):
            _, idx = hq.heappop(obstacle_distance_pq)
            k_nn_x_obstacles_idx.append(idx)
        
        return k_nn_x_obstacles_idx
    
    def find_obstacles_within_cbf_sensing_range(self, x_current, x_obstacles):
        obstacles_idx = []

        for i in range(len(x_obstacles)):
            distance = math.hypot(x_current[0] - x_obstacles[i][0], x_current[1] - x_obstacles[i][1])
            if distance <= self.cbf_constraints_sensing_radius:
                obstacles_idx.append(i)

        return obstacles_idx


    def motion_planning_with_QP(self,u_ref, model="linear_velocity_control"):
        x_current = self.y0
        x = np.zeros((2,0))
        u = np.zeros((2,0))
        delta_t = self.T/self.N
        time_step = 8
        for _ in range(time_step):
            for i in range(0,self.N):
                x=np.hstack((x,x_current))
                u_current = self.QP_controller(x_current[:,0],u_ref)
                u=np.hstack((u,u_current))
                x_current=x_current+delta_t*u_current
        return (x,u)

    def QP_constraint(self, x_current, u_ref, system_type="linear_velocity_control"):
        if system_type == "linear_velocity_control":
            x1 = x_current[0]
            x2 = x_current[1]
            
            u1_ref = u_ref[0] # Linear Velcoity Control
            u2_ref = u_ref[1] # Linear Velcoity Control
            # Control Acutuator Constraints
            u1_ref = np.clip(u1_ref, self.u1_lower_lim, self.u1_upper_lim)
            u2_ref = np.clip(u2_ref, self.u2_lower_lim, self.u1_upper_lim)
        
            obstacle_index = self.find_obstacles_within_cbf_sensing_range(x_current, self.x_obstacle)

            if obstacle_index:
                minCBF = float('inf')

                for index in obstacle_index:
                    h = (x1-self.x_obstacle[index][0])**2+(x2-self.x_obstacle[index][1])**2-self.x_obstacle[index][2]**2
                    lghu = 2*(x1-self.x_obstacle[index][0])*u1_ref+2*(x2-self.x_obstacle[index][1])*u2_ref
                    CBF_Constraint = lghu + 1.5 * self.k_cbf * h**self.p_cbf 
                    if CBF_Constraint < minCBF:
                        minCBF = CBF_Constraint
            
                if minCBF < 0:
                    return False

        elif system_type == "linear_acceleration_control":
            # print('acceleration')

            x1 = x_current[0]
            v1 = x_current[1]
            x2 = x_current[2]
            v2 = x_current[3]

            u1_ref = u_ref[0] # Linear Acceleration Control
            u2_ref = u_ref[1] # Linear Acceleration Control

            # Control Acutuator Constraints
            u1_ref = np.clip(u1_ref, self.u1_lower_lim, self.u1_upper_lim)
            u2_ref = np.clip(u2_ref, self.u2_lower_lim, self.u1_upper_lim)

            obstacle_index = self.find_obstacles_within_cbf_sensing_range(x_current, self.x_obstacle)

            if obstacle_index:
                minCBF = float('inf')
                for index in obstacle_index:
                    h = (x1-self.x_obstacle[index][0])**2+(x2-self.x_obstacle[index][1])**2-self.x_obstacle[index][2]**2
                    lfh = 2*(x1-self.x_obstacle[index][0])*v1+2*(x2-self.x_obstacle[index][1])*v2
                    lflfh = 2*v1**2+2*v2**2
                    lglfhu = 2*(x1-self.x_obstacle[index][0])*u1_ref+2*(x2-self.x_obstacle[index][1])*u2_ref
                    CBF_Constraint = lflfh + lglfhu + 3.0 * self.k_cbf*h + 1.5 * self.k_cbf2*lfh
                    if CBF_Constraint < minCBF:
                        minCBF = CBF_Constraint
                
                if minCBF < 0:
                    return False
                
        elif system_type == "unicycle_velocity_control":
            x = x_current[0]
            y = x_current[1]
            theta = x_current[2]
            v = u_ref[0]   #self.unicycle_constant_v # Fixed Linear Velocity

            w = u_ref[1] # Angular Velocity Control

            # States: x, y, theta
            obstacle_index = self.find_obstacles_within_cbf_sensing_range(x_current, self.x_obstacle)
            
            if obstacle_index:
                minCBF = float('inf')

                for index in obstacle_index:
                    xo = self.x_obstacle[index][0]
                    yo = self.x_obstacle[index][1]
                    r = self.x_obstacle[index][2]

                    # Unicycle with velocity control
                    h = (x-xo)**2+(y-yo)**2-r**2
                    Lfh = v*math.cos(theta)*(2*x-2*xo)+v*math.sin(theta)*(2*y-2*yo)
                    LfLfh = 2*(v**2)*math.cos(theta)**2+2*(v**2)*math.sin(theta)**2
                    LgLfh = v*math.cos(theta)*(2*y-2*yo) - v*math.sin(theta)*(2*x-2*xo)
            
                    CBF_Constraint = LfLfh+LgLfh*w+self.k1_unicyle_cbf*h+self.k2_unicyle_cbf*Lfh

                    if CBF_Constraint < minCBF:
                        minCBF = CBF_Constraint

                if minCBF < 0:
                    return False
                
        elif system_type == "unicycle_acceleration_control":
            # States: x, y, theta, v
            x = x_current[0]
            y = x_current[1]
            v = x_current[2]
            theta = x_current[3]

            u1 = u_ref[1] # Angular Velocity Control
            u2 = u_ref[0] # Linear Acceleration Control

            obstacle_index = self.find_obstacles_within_cbf_sensing_range(x_current, self.x_obstacle)
            if obstacle_index:
                minCBF = float('inf')

                for index in obstacle_index:
                    xo = self.x_obstacle[index][0]
                    yo = self.x_obstacle[index][1]
                    r = self.x_obstacle[index][2]

                    h = (x-xo)**2+(y-yo)**2-r**2
                    h_dot = v*(2*x - 2*xo)*math.cos(theta) + v*(2*y - 2*yo)*math.sin(theta)
                    h_dot_dot = u1*(-v*(2*x - 2*xo)*math.sin(theta) + v*(2*y - 2*yo)*math.cos(theta))
                    + u2*((2*x - 2*xo)*math.cos(theta) + (2*y - 2*yo)*math.sin(theta)) + 2*v**2*math.sin(theta)**2
                    + 2*v**2*math.cos(theta)**2

                    CBF_Constraint = h_dot_dot + 2*h*h_dot + (h_dot+h**2)**2

                    if CBF_Constraint < minCBF:
                        minCBF = CBF_Constraint
                if minCBF < 0:
                    return False

        return True


    def motion_planning_without_QP(self,u_ref,model="linear_velocity_control"):
        # model == "linear_velocity_control" or "linear_acceleration_control" 
        # or "unicycle_velocity_control" or "unicycle_acceleration_control"

        if model == "linear_velocity_control":
            # State: [position_x, position_y]^T, Control:[velocity_x, velocity_y]^T
            x_current = self.y0
            self.x = np.zeros((2,0))
            self.u = np.zeros((2,0))
            u_ref = np.array([[u_ref[0]], [u_ref[1]]])
            delta_t = self.T/self.N
            time_step = 8
            for _ in range(time_step):
                for i in range(0,self.N):
                    self.x = np.hstack((self.x, x_current))
                    self.u = np.hstack((self.u, u_ref))
                    if not self.QP_constraint(x_current[:,0],u_ref,system_type="linear_velocity_control"):
                        return (self.x, self.u)
                    x_current=x_current+delta_t*u_ref

            return (self.x,self.u)
        
        if model == "linear_acceleration_control":
            # State: [position_x, velocity_x, position_y, velocity_y]^T, Control:[acceleration_x, acceleration_y]^T
            x_current = self.y0
            p_current = [self.y0[0],self.y0[2]]
            v_current = [self.y0[1],self.y0[3]]

            self.x = np.zeros((4,0))
            self.u = np.zeros((2,0))
            u_ref = np.array([[u_ref[0]], [u_ref[1]]])
            delta_t = self.T/self.N
            time_step = 8
            for _ in range(time_step):
                for i in range(0,self.N):
                    self.x = np.hstack((self.x, x_current))
                    self.u = np.hstack((self.u, u_ref))
                    if not self.QP_constraint(x_current[:,0],u_ref,system_type="linear_acceleration_control"):
                        return (self.x, self.u)
                    #x_current=x_current+0.5*delta_t**2*u_ref
                    x_current[0] = p_current[0] + v_current[0]*delta_t + 0.5*u_ref[0]*delta_t**2
                    x_current[1] = v_current[0] + u_ref[0]*delta_t
                    x_current[2] = p_current[1] + v_current[1]*delta_t + 0.5*u_ref[1]*delta_t**2
                    x_current[3] = v_current[1] + u_ref[1]*delta_t

            return (self.x,self.u)
        
        elif model == "unicycle_velocity_control":
            # State: [position_x, position_y, theta] Control: [ Linear Velocity]
            x_current = self.y0
            self.x = np.zeros((3, 0))
            self.u = np.zeros((1, 0))
            u_ref = np.array(u_ref)

            def unicycle_model_velocity_control(t,y_input):
                return [math.cos(y_input[2])*self.unicycle_constant_v,math.sin(y_input[2])*self.unicycle_constant_v,u_ref]
            
            
            #if not self.QP_constraint_unicycle(x_current[:,0],u_ref):
            if not self.QP_constraint(x_current[:,0],u_ref,system_type="unicycle_velocity_control"):
                self.x = np.hstack((self.x, x_current))
                self.u = np.append(self.u, u_ref)
                return (self.x, self.u)
            else:
                solution = solve_ivp(fun=lambda t,y: unicycle_model_velocity_control(t,y), t_span = [0,self.T], y0=x_current[:,0],dense_output = True)
                self.x = np.hstack((self.x, np.array([solution.y[0],solution.y[1],solution.y[2]])))
                self.u = np.append(self.u, u_ref)
            
            return (self.x, self.u)
            
        elif model == "unicycle_acceleration_control":
            # State: [position_x, position_y, theta, velocity] Control: [Angular velocity, Linear Acceleration ]
            x_current = self.y0
            self.x = np.zeros((4, 0))
            self.u = np.zeros((2, 0))
            self.u_ref = np.array([[self.u_ref[0]], [self.u_ref[1]]])

            def CBF_constraint_checking_event(t, state):
                x_current = state
                if self.QP_constraint(x_current[:, 0], self.u_ref, system_type="unicycle_acceleration_control"):
                    return False # Continue solve_ivp because the control passes CBF constraints test
                else:
                    return True # Terminate solve_ivp because control is unsafe

            def unicycle_model(t, state):
                # The safe control is applied here, the function generates safe expanding edge (state trajectory)
                theta = state[2]
                v = state[3]

                dxdt = v*np.cos(theta)
                dydt = v*np.sin(theta)
                dthetadt = self.u_ref[0]
                dvdt = self.u_ref[1]

                return np.array([dxdt,dydt,dthetadt,dvdt])

            solution = solve_ivp(fun=unicycle_model,t_span=[0,self.T], y0=self.y0, events=CBF_constraint_checking_event)

            x_new_traj = solution.y # Return the state trajectory

            self.x = np.hsatck((self.x, x_new_traj))
            self.u = np.hstack((self.u, u_ref))

            return (self.x,self.u)


    def plot_traj(self,x,u):
        #t_span = np.linspace(0,self.T,self.N)

        fig, ax = plt.subplots()

        circle = plt.Circle((obstacle_list[0][0], obstacle_list[0][1]),
        obstacle_list[0][2], color='r',alpha=0.2)
        ax.add_artist(circle)
        ax.plot(x[0,:], x[1,:])
        ax.set_xlim(-1,5)
        ax.set_ylim(-1,5)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        plt.show()

if __name__ == "__main__":
    initial_state = np.array([[1.0], [1.0]])
    obstacle_list = [[2.9, 2.6, 0.5]]
    u_ref = [0.5, 0.5]

    CBFRRT_Planning = CBF_RRT(initial_state, obstacle_list)
    x, u= CBFRRT_Planning.motion_planning_without_QP(u_ref,model="linear_velocity_control")
    CBFRRT_Planning.plot_traj(x,u)
