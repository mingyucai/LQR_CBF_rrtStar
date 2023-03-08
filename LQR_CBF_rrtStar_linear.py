import os
import sys
import math
import numpy as np

import env, plotting, utils, Queue

from LQR_planning import LQRPlanner

import time

"""
LQR_CBF_RRT_star 2D
@author: mingyu cai

CBF_QP constraint is incorporated with LQR constraint
"""

class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None
        self.cost = 0


class LQRrrtStar:
    def __init__(self, x_start, x_goal, step_len,
                 goal_sample_rate, search_radius, iter_max):
        self.s_start = Node(x_start)
        self.s_goal = Node(x_goal)
        self.step_len = step_len
        self.goal_len = 8
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.iter_max = iter_max
        self.vertex = [self.s_start]
        self.path = []

        self.env = env.Env()
        self.plotting = plotting.Plotting(x_start, x_goal)
        self.utils = utils.Utils()

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

        self.lqr_planner = LQRPlanner()

    def planning(self):
        start_time = time.time()
        for k in range(self.iter_max):
            node_rand = self.generate_random_node(self.goal_sample_rate)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.LQR_steer(node_near, node_rand)


            if k % 100 == 0:
                print('rrtStar sampling iterations: ', k)
                print('rrtStar 1000 iterations sampling time: ', time.time() - start_time)
                start_time = time.time()

            if k % 2000 == 0:
                print('rrtStar sampling iterations: ', k)
                self.plotting.animation_online(self.vertex, "rrtStar", True)

            if node_new and not self.utils.is_collision(node_near, node_new):
                neighbor_index = self.find_near_neighbor(node_new)
                self.vertex.append(node_new)

                if neighbor_index:
                    self.LQR_choose_parent(node_new, neighbor_index)
                    self.rewire(node_new, neighbor_index)

        index = self.search_goal_parent()

        if index is None:
            print('No path found!')
            return None

        self.path = self.extract_path(self.vertex[index])

        print("path ", self.path)

        self.plotting.animation(self.vertex, self.path, "rrt*, N = " + str(self.iter_max))

    def sample_path(self, wx, wy, step=0.2):
        # smooth path
        px, py, traj_costs = [], [], []

        for i in range(len(wx) - 1):
            for t in np.arange(0.0, 1.0, step):
                px.append(t * wx[i+1] + (1.0 - t) * wx[i])
                py.append(t * wy[i+1] + (1.0 - t) * wy[i])

        dx, dy = np.diff(px), np.diff(py)
        traj_costs = [math.sqrt(idx ** 2 + idy ** 2) for (idx, idy) in zip(dx, dy)]
        return px, py, traj_costs

    def LQR_steer(self, node_start, node_goal):
        ##balance the distance of node_goal
        dist, theta = self.get_distance_and_angle(node_start, node_goal)
        dist = min(self.step_len, dist)
        node_goal.x = node_start.x + dist * math.cos(theta)
        node_goal.y = node_start.y + dist * math.sin(theta)

        wx, wy, _, _ = self.lqr_planner.lqr_planning(node_start.x, node_start.y, node_goal.x, node_goal.y, show_animation=False)
        px, py, traj_cost = self.sample_path(wx, wy)

        if len(wx) == 1:
            return None
        node_new = Node((wx[-1], wy[-1]))
        node_new.parent = node_start
        # calculate cost of each new_node
        node_new.cost = node_start.cost + sum(abs(c) for c in traj_cost)

        return node_new

    def cal_LQR_new_cost(self, node_start, node_goal):
        wx, wy, _, can_reach = self.lqr_planner.lqr_planning(node_start.x, node_start.y, node_goal.x, node_goal.y, show_animation=False)
        px, py, traj_cost = self.sample_path(wx, wy)
        if wx is None:
            return float('inf'), False
        return node_start.cost + sum(abs(c) for c in traj_cost), can_reach

    def LQR_choose_parent(self, node_new, neighbor_index):
        cost = []
        for i in neighbor_index:

            # check if neighbor_node can reach node_new
            _, _, _, can_reach = self.lqr_planner.lqr_planning(self.vertex[i].x, self.vertex[i].y, node_new.x, node_new.y, show_animation=False)

            if can_reach and not self.utils.is_collision(self.vertex[i], node_new):  #collision check should be updated if using CBF
                update_cost, _ = self.cal_LQR_new_cost(self.vertex[i], node_new)
                cost.append(update_cost)
            else:
                cost.append(float('inf'))
        min_cost = min(cost)

        if min_cost == float('inf'):
            print('There is no good path.(min_cost is inf)')
            return None

        cost_min_index = neighbor_index[int(np.argmin(cost))]
        node_new.parent = self.vertex[cost_min_index]


    def rewire(self, node_new, neighbor_index):
        for i in neighbor_index:
            node_neighbor = self.vertex[i]

            # check collision and LQR reachabilty
            if not self.utils.is_collision(node_new, node_neighbor):
                new_cost, can_rach = self.cal_LQR_new_cost(node_new, node_neighbor)

                if can_rach and node_neighbor.cost > new_cost:
                    node_neighbor.parent = node_new
                    node_neighbor.cost = new_cost


    def search_goal_parent(self):
        print('goal ', self.s_goal.x, self.s_goal.y)
        dist_list = [math.hypot(n.x - self.s_goal.x, n.y - self.s_goal.y) for n in self.vertex]
        node_index = [i for i in range(len(dist_list)) if dist_list[i] <= self.goal_len]

        if not node_index:
            return None

        if len(node_index) > 0:
            cost_list = [dist_list[i] + self.vertex[i].cost for i in node_index
                         if not self.utils.is_collision(self.vertex[i], self.s_goal)]
            print('found destination ', node_index[int(np.argmin(cost_list))])
            print('minimum vetex dis to goal ', min(cost_list))
            print("min distance", min(dist_list))
            return node_index[int(np.argmin(cost_list))]

        return len(self.vertex) - 1


    def generate_random_node(self, goal_sample_rate):
        delta = self.utils.delta

        if np.random.random() > goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return self.s_goal

    def find_near_neighbor(self, node_new):
        n = len(self.vertex) + 1
        r = min(self.search_radius * math.sqrt((math.log(n) / n)), self.step_len)

        dist_table = [math.hypot(nd.x - node_new.x, nd.y - node_new.y) for nd in self.vertex]
        dist_table_index = [ind for ind in range(len(dist_table)) if dist_table[ind] <= r and
                            not self.utils.is_collision(node_new, self.vertex[ind])]
        return dist_table_index

    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]


    def extract_path(self, node_end):
        path = [[self.s_goal.x, self.s_goal.y]]
        print('initial path', path)
        node = node_end

        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)


def main():
    # x_start = (18, 8)  # Starting node
    # x_goal = (37, 18)  # Goal node
    x_start = (2, 2)  # Starting node
    x_goal = (30, 24)  # Goal node


    rrt_star = LQRrrtStar(x_start, x_goal, 10, 0.10, 20, 6000)
    rrt_star.planning()


if __name__ == '__main__':
    main()