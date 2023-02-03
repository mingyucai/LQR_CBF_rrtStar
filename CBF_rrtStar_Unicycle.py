"""
RRT_star 2D
@author: mingyu cai
"""

import os
import random
import sys
import math
import numpy as np

import env, plotting, utils
from CBFsteer import CBF_RRT

class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.theta = n[2]
        self.parent = None

class Rrt:
    def __init__(self, s_start, s_goal, step_len, goal_sample_rate, search_radius, iter_max, QP = False):
        self.s_start = Node(s_start)
        self.s_goal = Node(s_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.iter_max = iter_max
        self.vertex = [self.s_start]

        self.env = env.Env()
        self.plotting = plotting.Plotting(s_start, s_goal)
        self.utils = utils.Utils()

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary
        self.u_ref_nominal = 2.
        self.sovle_QP =QP

    def planning(self):
        for i in range(self.iter_max):
            node_rand = self.generate_random_node(self.goal_sample_rate)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new, desired_theta = self.new_state(node_near, node_rand)

            #sampled_theta = random.gauss(desired_theta, 0.2)
            #u_ref = [self.u_ref_nominal * math.cos(sampled_theta), self.u_ref_nominal * math.sin(sampled_theta)]
            
            if not self.sovle_QP:
                cbf_rrt_simulation = CBF_RRT(np.array([[node_near.x], [node_near.y],[node_near.theta]]), self.obs_circle)
                #CLF QP to compute reference control (w) to converge to new node
                u_ref = cbf_rrt_simulation.CLF_unicycle_QP([node_near.x, node_near.y, node_near.theta],[node_new.x,node_new.y,0])
                x_simulated, u_simulated = cbf_rrt_simulation.motion_planning_without_QP(u_ref, model="unicycle")
                feasible = True
                node_new.x = x_simulated[0][-1]
                node_new.y = x_simulated[1][-1]
                node_new.theta = x_simulated[2][-1]

            else:
                try:
                    cbf_rrt_simulation = CBF_RRT(np.array([[node_near.x],[node_near.y]]), self.obs_circle, model="unicycle")
                    x_simulated, u_simulated= cbf_rrt_simulation.motion_planning_with_QP(u_ref, model="unicycle")
                    
                    print('try', x_simulated)
                    feasible = True
                    node_new.x = x_simulated[0][-1]
                    node_new.y = x_simulated[1][-1]
                except:
                    feasible = False

            if i % 200 == 0:
                print('CBF_rrtStart iterations: ', i)

            if i % 1000 == 0:
                self.plotting.animation_online(self.vertex, "RRT", True)

            # feasible = True
            if feasible:
                if node_new and not self.utils.is_collision(node_near, node_new):
                    neighbor_index = self.find_near_neighbor(node_new)
                    self.vertex.append(node_new)

                    if neighbor_index:
                        self.choose_parent(node_new, neighbor_index)
                        self.rewire(node_new, neighbor_index)
        index = self.search_goal_parent()
        self.path = self.extract_path(self.vertex[index])
        self.plotting.animation(self.vertex, self.path, "rrt*, N = " + str(self.iter_max))
        if self.path:
            return self.path
        return None

    def generate_random_node(self, goal_sample_rate):
        delta = self.utils.delta

        if np.random.random() > goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta),
                         0))

        return self.s_goal

    def find_near_neighbor(self, node_new):
        n = len(self.vertex) + 1
        r = min(self.search_radius * math.sqrt((math.log(n) / n)), self.step_len)

        dist_table = [math.hypot(nd.x - node_new.x, nd.y - node_new.y) for nd in self.vertex]
        dist_table_index = [ind for ind in range(len(dist_table)) if dist_table[ind] <= r and
                            not self.utils.is_collision(node_new, self.vertex[ind])]
        return dist_table_index

    def choose_parent(self, node_new, neighbor_index):
        cost = [self.get_new_cost(self.vertex[i], node_new) for i in neighbor_index]

        cost_min_index = neighbor_index[int(np.argmin(cost))]
        node_new.parent = self.vertex[cost_min_index]

    def rewire(self, node_new, neighbor_index):
        for i in neighbor_index:
            node_neighbor = self.vertex[i]
            if self.cost(node_neighbor) > self.get_new_cost(node_new, node_neighbor):
                node_neighbor.parent = node_new

    def search_goal_parent(self):
        dist_list = [math.hypot(n.x - self.s_goal.x, n.y - self.s_goal.y) for n in self.vertex]
        node_index = [i for i in range(len(dist_list)) if dist_list[i] <= self.step_len]

        if len(node_index) > 0:
            cost_list = [dist_list[i] + self.cost(self.vertex[i]) for i in node_index
                         if not self.utils.is_collision(self.vertex[i], self.s_goal)]
            return node_index[int(np.argmin(cost_list))]

        return len(self.vertex) - 1

    def get_new_cost(self, node_start, node_end):
        dist, _ = self.get_distance_and_angle(node_start, node_end)
        return self.cost(node_start) + dist


    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]

    @staticmethod
    def cost(node_p):
        node = node_p
        cost = 0.0

        while node.parent:
            cost += math.hypot(node.x - node.parent.x, node.y - node.parent.y)
            node = node.parent

        return cost

    def new_state(self, node_start, node_end):
        dist, theta = self.get_distance_and_angle(node_start, node_end)

        dist = min(self.step_len, dist)

        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta),0))
        node_new.parent = node_start

        return node_new, theta

    def extract_path(self, node_end):
        path = [(self.s_goal.x, self.s_goal.y)]
        node_now = node_end

        while node_now.parent is not None:
            node_now = node_now.parent
            path.append((node_now.x, node_now.y))

        return path

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)


def main():
    x_start = (2, 2,0)  # Starting node
    x_goal = (49, 24,0)  # Goal node

    solve_QP = False
    rrt = Rrt(x_start, x_goal, 10., 0.05, 10, 5000, solve_QP)
    rrt.planning()

if __name__ == '__main__':
    main()