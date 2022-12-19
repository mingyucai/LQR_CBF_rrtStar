

import matplotlib.pyplot as plt
import math
import numpy as np
import heapq as hq
import sys

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
    def __init__(self, initial_state, obstacle_list):
        self.t0 = 0 # Starting time
        self.T = 0.2  #Integration Length
        self.N = 50 # Number of Control Updates
        self.y0 = initial_state
        self.k = 3 # k nearest neighbor obstacles that will be used for generating CBF constraint
        self.k_cbf = 1.0 #CBF coefficient
        self.p_cbf = 1 #CBF constraint power
        self.x_obstacle = obstacle_list
        self.u1_lower_lim = -5
        self.u1_upper_lim = 5
        self.u2_lower_lim = -5
        self.u2_upper_lim = 5

        self.u1_traj = np.zeros(shape=(0,0))
        self.u2_traj = np.zeros(shape=(0,0))
        self.x1_traj = np.zeros(shape=(0,0))
        self.x2_traj = np.zeros(shape=(0,0))

        self.cbf_traj = np.zeros(shape=(0,0))
        self.hdot_traj = np.zeros(shape=(0,0))
        self.h_traj = np.zeros(shape=(0,0))


    def QP_controller(self,x_current,u_ref):
        self.m = Model("CBF_CLF_QP")
        x1 = x_current[0]
        x2 = x_current[1]

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
        '''
        h_temp = (x1-self.x_obstacle[i][0])**2+(x2-self.x_obstacle[i][1])**2-self.x_obstacle[i][2]**2
        lgh_temp = 2*(x1-self.x_obstacle[i][0])*u1+2*(x2-self.x_obstacle[i][1])*u2
        self.cbf_traj = np.append(self.cbf_traj,(lgh_temp+self.k_cbf*h_temp**self.p_cbf))
        self.hdot_traj = np.append(self.hdot_traj,(lgh_temp))
        self.h_traj = np.append(self.h_traj,(self.k_cbf*h_temp**1))
        '''
        return np.array([[u1],[u2]])

    def find_knn_obstacle(self, x_current, x_obstacles,k):
        '''
            Take in current node's position and obstacles list and integer k
            return k obstacles index based on distance (from close to far)
        '''
        try:
            assert k <= len(x_obstacles)
        except:
            print("The integer k must be smaller than total number of obstacles!")
            sys.exit(1)

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

    def motion_planning_with_QP(self,u_ref):
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

    def QP_constraint(self,x_current,u_ref):
        # self.m = Model("CBF_CLF_QP")
        x1 = x_current[0]
        x2 = x_current[1]
        # self.m.remove(self.m.getConstrs())
        u1_ref = u_ref[0]
        u2_ref = u_ref[1]
        # Control Acutuator Constraints
        u1_ref = np.clip(u1_ref, self.u1_lower_lim, self.u1_upper_lim)
        u2_ref = np.clip(u2_ref, self.u2_lower_lim, self.u1_upper_lim)
        
        knn_obstacle_index = self.find_knn_obstacle(x_current, self.x_obstacle, self.k)

        # check CBF constraint
        for index in knn_obstacle_index:
            h = (x1-self.x_obstacle[index][0])**2+(x2-self.x_obstacle[index][1])**2-self.x_obstacle[index][2]**2

            lgh = 2*(x1-self.x_obstacle[index][0])*u1_ref+2*(x2-self.x_obstacle[index][1])*u2_ref
            CBF_Constraint = lgh+self.k_cbf*h**self.p_cbf >= 0
            if not CBF_Constraint:
                return False
        return True


    def motion_planning_without_QP(self,u_ref):
        x_current = self.y0
        x = np.zeros((2,0))
        u = np.zeros((2,0))
        u_ref = np.array([[u_ref[0]], [u_ref[1]]])
        delta_t = self.T/self.N
        time_step = 8
        for _ in range(time_step):
            for i in range(0,self.N):
                x=np.hstack((x,x_current))
                u = np.hstack((u, u_ref))
                if not self.QP_constraint(x_current[:,0],u_ref):
                    return (x, u)
                x_current=x_current+delta_t*u_ref
        return (x,u)

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
    initial_state = np.array([[1.0],[1.0]])
    obstacle_list = [[2.9,2.6,0.5]]
    u_ref = [0.5,0.5]

    CBFRRT_Planning = CBF_RRT(initial_state, obstacle_list)
    x, u= CBFRRT_Planning.motion_planning(u_ref)
    CBFRRT_Planning.plot_traj(x,u)