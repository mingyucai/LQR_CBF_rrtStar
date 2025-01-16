import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
import env, plotting, utils
from LQR_planning import LQRPlanner_acceleration

import copy
import time

class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.vx = 0
        self.vy = 0
        self.parent = None
        self.cost = 0
        self.StateTraj = None # For CEM sampling
        self.state_parent_to_current = None # State with interpolated points
        self.u_parent_to_current = None # Acceleration
        self.childrenNodeInds = set([])

class LQRrrtStar:
    def __init__(self, x_start, x_goal, step_len,
                 goal_sample_rate, search_radius, iter_max,AdSamplingFlag = False):
        self.s_start = Node(x_start)
        self.s_goal = Node(x_goal)
        self.start_state = x_start
        self.step_len = step_len
        self.goal_len = 8
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.iter_max = iter_max
        self.vertex = [self.s_start]


        self.env = env.Env()
        self.plotting = plotting.Plotting(x_start, x_goal)
        self.utils = utils.Utils()

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

        self.lqr_planner = LQRPlanner_acceleration()

        # The adaptive sampling attributes: 
        self.Vg_leaves = []
        self.AdSamplingFlag = AdSamplingFlag
        self.adapIter = 1
        self.kde_preSamples = []
        self.kde_currSamples =[]
        self.initEliteSamples = []
        self.curr_Ldist = 0
        self.prev_Ldist = 0
        #Reaching the optimal distribution params:
        #---kde
        self.kdeOpt_flag = False
        self.kde_eliteSamples = []
        self.KDE_fitSamples = None
        self.KDE_pre_gridProbs = None
        self.kde_enabled = False
        #Elite samples and CE computation att
        self.len_frakX = 0
        self.pre_gridProbs = []
        self.SDF_optFlg = False
        self.N_qSamples = 200 
        self.rho = .3
        self.step_size = 0.3
        self.plot_pdf_kde = False

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

            if k % 500 == 0:
                print('rrtStar sampling iterations: ', k)
                self.plotting.animation_online(self.vertex, "rrtStar", True)

            if node_new and not self.utils.is_collision(node_near, node_new):
                neighbor_index = self.find_near_neighbor(node_new)
                self.vertex.append(node_new)

                if neighbor_index:
                    self.LQR_choose_parent(node_new, neighbor_index)
                    self.rewire(node_new, neighbor_index)
            
            # >>> Extend to the goal 
            if self.AdSamplingFlag: 
                # Steering to the goal region: 
                if node_new is None: 
                    continue
                g_node = self.LQR_steer(node_new, self.s_goal, exact_steering = False)
                if g_node is not None and not self.utils.is_collision(node_new, g_node):
                    self.Vg_leaves.append(g_node)
            # <<< End extend to the goal
        
        index = self.search_goal_parent()

        if index is None:
            print('No path found!')
            return None

        self.path, path_continuous, control_path_list = self.extract_path(self.vertex[index])
        self.save_state_and_control_trajectory_as_numpy(self.path, control_path_list)

        '''
        
        # For visualizing if the state trajectory is correct through open-loop control
        simulated_state_traj = self.utils.integrate_double_integrator(x_init=np.array([self.start_state[0],
                                                                                       0,
                                                                                       self.start_state[1],
                                                                                       0]),
                                                                                       u=control_path_list, dt=0.05)
        plt.title("State trajectory via Open-loop control")
        plt.plot(simulated_state_traj[:, 0], simulated_state_traj[:, 2], 'r--')
        plt.xlim([self.x_range[0], self.x_range[1]])
        plt.ylim([self.y_range[0], self.y_range[1]])
        self.plotting.animation(self.vertex, self.path, "rrt*, N = " + str(self.iter_max))
        '''
    def sample_path(self, wx, wy, u_sequence, step=0.2):
        # smooth path
        px, py, traj_costs = [], [], []

        # Convert u_sequence with matrix form to list with float numbers
        u_sequence_list = [[u_sequence[i].item(0, 0), u_sequence[i].item(1, 0)] for i in range(len(u_sequence))]

        for i in range(len(wx) - 1):
            for t in np.arange(0.0, 1.0, step):
                px.append(t * wx[i+1] + (1.0 - t) * wx[i])
                py.append(t * wy[i+1] + (1.0 - t) * wy[i])

        dx, dy = np.diff(px), np.diff(py)
        u_sequence_cost = sum([np.linalg.norm(u) for u in u_sequence_list])
        traj_costs = [math.sqrt(idx ** 2 + idy ** 2) for (idx, idy) in zip(dx, dy)]
        return px, py, traj_costs, u_sequence_cost
    

    def LQR_steer(self, node_start, node_goal,exact_steering = False):
        ##balance the distance of node_goal
        dist, theta = self.get_distance_and_angle(node_start, node_goal)
        if not exact_steering: 
            dist = min(self.step_len, dist)
            show_animation = False
        else: 
            show_animation = False
        node_goal.x = node_start.x + dist * math.cos(theta)
        node_goal.y = node_start.y + dist * math.sin(theta)

        wx, wy, _, _, u_sequence = self.lqr_planner.lqr_planning(node_start.x, node_start.y, node_start.vx,
                                                                 node_start.vy, node_goal.x, node_goal.y,
                                                                 node_goal.vx, node_goal.vy,
                                                                 show_animation=show_animation,cbf_check=True)

        # This is for adapative sampling
        px, py, traj_cost, u_sequence_cost = self.sample_path(wx, wy, u_sequence)

        if len(wx) == 1:
            return None
        node_new = Node((wx[-1], wy[-1]))

        node_new.parent = node_start
        # calculate cost of each new_node
        node_new.cost = node_start.cost + sum(abs(c) for c in traj_cost) + u_sequence_cost
        node_new.StateTraj = np.array([px,py]) # Will be needed for adaptive sampling
        node_new.state_parent_to_current = [wx, wy]
        node_new.u_parent_to_current = u_sequence
        return node_new

    def cal_LQR_new_cost(self, node_start, node_goal,cbf_check = True):
        wx, wy, _, can_reach, u_sequence = self.lqr_planner.lqr_planning(node_start.x, node_start.y, node_start.vx,
                                                                         node_start.vy, node_goal.x, node_goal.y,
                                                                         node_goal.vx, node_goal.vy,
                                                                         show_animation=False,cbf_check = cbf_check)
        px, py, traj_cost, u_sequence_cost = self.sample_path(wx, wy, u_sequence) # TO DO: update LQR cost
        if wx is None:
            return float('inf'), False
        return node_start.cost + sum(abs(c) for c in traj_cost)+u_sequence_cost, can_reach, u_sequence

    def LQR_choose_parent(self, node_new, neighbor_index):
        cost = []
        u_neighbor = []  # store u_sequence of neighbor nodes
        for i in neighbor_index:

            # check if neighbor_node can reach node_new
            _, _, _, can_reach, u_sequence = self.lqr_planner.lqr_planning(self.vertex[i].x, self.vertex[i].y,
                                                                           self.vertex[i].vx, self.vertex[i].vy,
                                                                           node_new.x, node_new.y, node_new.vx,
                                                                           node_new.vy, show_animation=False)

            if can_reach and not self.utils.is_collision(self.vertex[i], node_new):
                update_cost, _, u_sequence = self.cal_LQR_new_cost(self.vertex[i], node_new)
                u_sequence.reverse()
                cost.append(update_cost)
                u_neighbor.append(u_sequence)
            else:
                cost.append(float('inf'))
                u_neighbor.append(None)
        min_cost = min(cost)

        if min_cost == float('inf'):
            print('There is no good path.(min_cost is inf)')
            return None

        neighbor_index_with_minimum_cost = np.argmin(cost)
        cost_min_index = neighbor_index[neighbor_index_with_minimum_cost]
        node_new.parent = self.vertex[cost_min_index]
        node_new.u_parent_to_current = u_neighbor[neighbor_index_with_minimum_cost]
        # Add the index of node_new to the children of its parent. This step is essential when rewiring the tree to
        # project the changes of the cost of the rewired node to its antecessors
        node_new.parent.childrenNodeInds.add(len(self.vertex)-1)
        

    def rewire(self, node_new, neighbor_index):
        for i in neighbor_index:
            node_neighbor = self.vertex[i]

            # check collision and LQR reachabilty
            if not self.utils.is_collision(node_new, node_neighbor):
                new_cost, can_rach, u_sequence = self.cal_LQR_new_cost(node_new, node_neighbor)

                if can_rach and node_neighbor.cost > new_cost:
                    u_sequence.reverse()
                    node_neighbor.parent = node_new
                    node_neighbor.cost = new_cost
                    node_neighbor.u_parent_to_current = u_sequence
                    self.updateCosts(node_neighbor)

    def updateCosts(self,node):
        for ich in node.childrenNodeInds: 
            self.vertex[ich].cost = self.cal_LQR_new_cost(node,self.vertex[ich],cbf_check = False)[0]
            self.updateCosts(self.vertex[ich])

    def search_goal_parent(self):
        dist_list = [math.hypot(n.x - self.s_goal.x, n.y - self.s_goal.y) for n in self.vertex]
        node_index = [i for i in range(len(dist_list)) if dist_list[i] <= self.goal_len]

        if not node_index:
            return None

        if len(node_index) > 0:
            cost_list = [dist_list[i] + self.vertex[i].cost for i in node_index
                         if not self.utils.is_collision(self.vertex[i], self.s_goal)]
            return node_index[int(np.argmin(cost_list))]

        return len(self.vertex) - 1


    def generate_random_node(self, goal_sample_rate,rce = 0.5,md = 8):
        delta = self.utils.delta
        adap_flag=self.AdSamplingFlag
        u_rand = np.random.uniform(0, 1)
        Vg_leaves = self.Vg_leaves
        if not adap_flag or (u_rand > rce and len(Vg_leaves) == 0): # Uniform sampling form the workspace:
            if np.random.random() > goal_sample_rate:
                return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                            np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

            return copy.deepcopy(self.s_goal)
        elif len(Vg_leaves) != 0 and not self.SDF_optFlg:
            N_xSmpls = 200 
            # t_min = min([vg.curTime for vg in Vg_leaves]) # The fastest trajectory 
            # h = t_min/md 
            h = .5
            return self.CE_Sample(Vg_leaves,h,self.N_qSamples)
        elif self.SDF_optFlg: # Sampling from the optimal SDF: 
            xySmpl = self.OptSDF.sample()
            return Node(xySmpl[0][0],xySmpl[0][1])
        else: 
            if np.random.random() > goal_sample_rate:
                return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                            np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))
            return copy.deepcopy(self.s_goal)

    def CE_Sample(self,Vg_leaves,h,N_qSamples):
        """
        Exploit the samples of trajectories that reach the goal to adapt the sampling distribution towards
        the distribution of the rare event.
        The trajectories that reach the goal will be disceretized to extract the elite samples that will be
        used to adapt (optimize)
        the sampling distribution.

        :param Vg_leaves:
        :param h: The time step to discretize the trajectories
        :param N_qSamples: A threshold indicates the number of samples that are sufficient enough to be exploited
        (TODO (Doc): How to decide this number)
        :return: None: if the number of points of the discretized trajectories < N_qSamples, (x,y) samples from the
        estimated distribution
        """
        if len(Vg_leaves)>=(self.adapIter*30): #The acceptable number of trajectories to adapat upon
            frakX = []
            #Find the elite trajectoies then discretize them and use their samples as the elite samples:
            Vg_leaves_costList = [vg.cost for vg in Vg_leaves]
            if (self.adapIter + 3) > 5: 
                d_factor = 15
            elif self.adapIter > 2:
                d_factor = self.adapIter + 3
            else: 
                d_factor = self.adapIter
            
            q = self.rho               # The rho^th quantile  
            cost_rhoth_q = np.quantile(Vg_leaves_costList, q=q)
            elite_Vg_leaves = [vg for vg in Vg_leaves if vg.cost <= cost_rhoth_q]
            if len(elite_Vg_leaves) == 0:
                elite_Vg_leaves = Vg_leaves

            #XXXXXXX
            for vg in elite_Vg_leaves:
                vgcost2come = vg.cost
                #Concatnating the trajectory:
                traj2vg = np.asarray(vg.StateTraj).T
                node = vg
                while node.parent is not None:
                    node = node.parent
                    if node.cost != 0:
                        ParentTraj = node.StateTraj.T
                        traj2vg = np.concatenate((ParentTraj,traj2vg),axis=0)
                # Backtrack the path from vg to v0; extract the sample at certain increments of the time:
                tStep_init1 = int(h/self.step_size)
                tStep_init = 2
                tStep = 10
                tStep_temp = tStep_init1+3
                while tStep < len(traj2vg[:,1]):
                    pi_q_tStep = traj2vg[tStep,:]
                    elite_cddtSample = [pi_q_tStep,vgcost2come] #This tuple contains the actual sample pi_q_tStep
                    # and the CostToCome to the goal of the corresponding trajectory
                    frakX.append(elite_cddtSample)
                    tStep = tStep + tStep_temp
            if self.adapIter == 1:
                frakX.extend(self.initEliteSamples)
            # XXXXXXX
            self.len_frakX = len(frakX)
            if len(frakX) == 0:
                ok = 1
            x,y = self.CE_KDE_Sampling(frakX)
        else:
                x = None
                y = None
        if x is None or y is None: 
            delta = self.utils.delta
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                            np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))
        else: 
            return Node((x,y))
    #Density estimate, kernel density or GMM:
    def CE_KDE_Sampling(self,frakX):
        """
        Fit the elite samples to Kernel density estimate (KDE) or a GMM to generate from; and generate an (x,y)
        sample from the estimated
        distribution. Checks if the CE between the previous density estimate and the current one below some threshold.
        In the case
        of KDE the expectation similarity measure could be used instead on the CE.

        NOTE to Ahmad:
        You're using the CE with the KDE because you have the logistic probes of the samples and you use them;
        however, for
        Kernel based distributions the expectation similarity could be used as well. One might reformulate the
        CE framework
        in terms of nonparametric distributions.

        :param frakX: The elite set of samples with the corresponding trajectory cost.
        :return:
        """
        frakXarr = np.array(frakX)
        N_samples = len(frakX)
        if len(frakXarr.shape) !=2:
            ok =1
        costs_arr = frakXarr[:,1]
        elite_samplesTemp = frakXarr[:,0] #A subset of the samples that are below the elite quantile
        elite_samples = [elite_samplesTemp[i] for i in range(len(elite_samplesTemp))]
        elite_samples_arr = np.asarray(elite_samples)
        elite_costs = costs_arr

        #random point from the estimated distribution:
        if self.kde_enabled:#self.params.kde_enabled:
            kde = KernelDensity(kernel='gaussian', bandwidth=.85)
            # kde.fit(elite_samples_arr,sample_weight=w_arrNorm)
            kde.fit(elite_samples_arr)
            self.adapIter += 1
            xySample = kde.sample()

        if self.kde_enabled:#self.params.kde_enabled:
            x_gridv = np.linspace(-2, 18, 40)
            y_gridv = np.linspace(-2, 18, 40)
            Xxgrid, Xygrid = np.meshgrid(x_gridv, y_gridv)
            XYgrid_mtx = np.array([Xxgrid.ravel(), Xygrid.ravel()]).T
            #Get the probabilities
            grid_probs = np.exp(kde.score_samples(XYgrid_mtx))

            # Find the KL divergence the current samples and the previous ones:
            if self.adapIter > 2:
                KL_div = self.KLdiv(grid_probs)
                if KL_div < .1:
                    self.kdeOpt_flag = True
                    
                self.KDE_fitSamples = kde

            self.KDE_pre_gridProbs = grid_probs

            #Plot the distribution
            if self.plot_pdf_kde:
                # self.initialize_graphPlot()
                # , norm=LogNorm(vmin=4.18, vmax=267.1))
                CS = plt.contour(Xxgrid, Xygrid, grid_probs.reshape(Xxgrid.shape))
                # plt.colorbar(CS, shrink=0.8, extend='both')
                plt.scatter(elite_samples_arr[:, 0], elite_samples_arr[:, 1])
                plt.show()

        return xySample[0][0],xySample[0][1]

    def KLdiv(self,grid_probs):
        """
        Compute the KL divergence
        :param grid_probs: The probabilities of the point in the grid of the current sampling distribution
        :return: the KL divergence
        """
        if self.kde_enabled:#self.params.kde_enabled:
            pre_grid_probs = self.KDE_pre_gridProbs
        else:
            pre_grid_probs = self.pre_gridProbs
        return -sum([pre_grid_probs[i]*np.log2(grid_probs[i]/pre_grid_probs[i]) for i in range(len(pre_grid_probs))])
    # End of Adaptive sampling 

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
        path_continuous = []
        path_continuous_formated = [[],[]]
        u_path = []
        node = node_end

        while node.parent is not None:
            path.append([node.x, node.y])
            u_path.extend(node.u_parent_to_current)
            path_continuous.extend([node.state_parent_to_current[0], node.state_parent_to_current[1]])
            node = node.parent
        path.append([node.x, node.y])

        u_path_list = [[u_path[i].item(0, 0), u_path[i].item(1, 0)] for i in range(len(u_path))]
        u_path_list.reverse()
        path_continuous.reverse()
        path.reverse()

        for i in range(len(path_continuous)):
            if i % 2 == 0:
                path_continuous_formated[0].extend(path_continuous[i])

        for i in range(len(path_continuous)):
            if i % 2 != 0:
                path_continuous_formated[1].extend(path_continuous[i])

        # Add dummy velocity profile
        zero_vel_list = [0.0] * len(path_continuous_formated[1])
        path_continuous_formated.insert(1, zero_vel_list)
        path_continuous_formated.insert(3, zero_vel_list)

        return path, path_continuous_formated, u_path_list

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    def save_state_and_control_trajectory_as_numpy(self, state_list, control_list):
        cwd = os.getcwd()
        os_path_for_state = os.path.join(cwd, 'output_state_control_trajs',
                                         'state_double_integrator_traj.npy')
        os_path_for_control = os.path.join(cwd, 'output_state_control_trajs',
                                           'control_double_integrator_traj.npy')
        print("Saving state and control trajectory...")

        np.save(os_path_for_control, np.array(control_list))
        np.save(os_path_for_state, np.array(state_list))
        print("Complete.")

def main():
    # x_start = (18, 8)  # Starting node
    # x_goal = (37, 18)  # Goal node
    x_start = (2.0, 2.0)  # Starting node
    x_goal = (30.0, 24.0)  # Goal node

    rrt_star = LQRrrtStar(x_start, x_goal, 10, 0.10, 20, 500, AdSamplingFlag = False)
    rrt_star.planning()


if __name__ == '__main__':
    main()
