from rrt_tools import RRT_tools
from kuka_sim import *
import random 
import time
import scipy.interpolate
import numpy as np
import matplotlib.pyplot as plt 
class RRTAnalysis():
    def __init__(self) -> None:
        self.start_to_goal_tree = None
        self.goal_to_start_tree = None
        self.running_rrt = None 
        self.running_rrt_connect = None
        self.time_log = [] 
        self.start_to_goal_tree_size_log = []
        self.goal_to_start_tree_size_log = []
        self.new_node_connection_distance = []

    def register_RRT_algo(self,start_to_goal_tree):
        self.start_to_goal_tree = start_to_goal_tree 
        self.running_rrt = True
        self.running_rrt_connect = False
    
    def register_RRT_Connect_algo(self,start_to_goal_tree, goal_to_start_tree):
        self.start_to_goal_tree = start_to_goal_tree
        self.goal_to_start_tree = goal_to_start_tree
        self.running_rrt_connect = True
        self.running_rrt = False 

    def log_connection_length(self, parent_node, child_node):
        c_parent = parent_node.value
        c_child = child_node.value
        l2_distance = np.linalg.norm(np.array(c_parent)-np.array(c_child))
        self.new_node_connection_distance.append(l2_distance)
    def monitor_RRT_size(self,):
        self.time_log.append(time.time())
        #instrospect start to goal tree
        start_to_goal_tree_size = len(self.start_to_goal_tree.rrt_tree.configurations_dict)
        #print(start_to_goal_tree_size)
        self.start_to_goal_tree_size_log.append(start_to_goal_tree_size)
        #if running rrt connect, introspect goal to start tree as well 
        if self.running_rrt_connect:
            goal_to_start_tree_size = len(self.goal_to_start_tree.rrt_tree.configurations_dict)
            #print(goal_to_start_tree_size)
            self.goal_to_start_tree_size_log.append(goal_to_start_tree_size)
    
    def plot_rrt_nodes_over_time(self,):
        pass 
    def plot_rrt_connection_length_over_time(self,):
        pass 
class KukaRRTPlanner():
    def __init__(self) -> None:
        is_visualizing = True 
        self.R_WG = RotationMatrix(np.array([[0,1,0], [1,0,0], [0,0,-1]]).T)
        #Goal outside shelf 
        #self.T_WG_goal = RigidTransform(p=np.array([4.69565839e-01, 2.95894043e-16, 0.65]), R=self.R_WG)
        #Goal inside shelf
        self.T_WG_goal = RigidTransform(p=np.array([8.19565839e-01, 2.95894043e-16, 0.65]), R=self.R_WG)
        self.meshcat = StartMeshcat()
        self.env = ManipulationStationSim(self.meshcat,is_visualizing)
        self.q_start = self.env.q0
        ik_solver = IKSolver()
        self.q_goal, optimal = ik_solver.solve(self.T_WG_goal, q_guess=self.q_start)
        self.gripper_setpoint = 0.1
        self.door_angle = np.pi/2 - 0.001
        self.left_door_angle = -np.pi/2
        self.right_door_angle = np.pi/2
        
        self.iiwa_problem = IiwaProblem(
            q_start=self.q_start,
            q_goal=self.q_goal,
            gripper_setpoint=self.gripper_setpoint,
            left_door_angle=self.left_door_angle,
            right_door_angle=self.right_door_angle,
            meshcat = self.meshcat,
            is_visualizing=is_visualizing)
        AddMeshcatTriad(self.meshcat, 'goal pose', X_PT=self.T_WG_goal, opacity=.5)
        self.env.DrawStation(self.q_start,self.gripper_setpoint,self.left_door_angle,self.right_door_angle)
    
    def _start_meshcat_vis(self):
        """
        Starts meshcat instance
        """
        # Start the visualizer.
        meshcat = StartMeshcat()
        print("loading meshcat...")
        time.sleep(2.5) #need to wait for meshcat server to initialize
        return meshcat 
    
    def rrt_extend_operation(self, new_q_sample, rrt: RRT_tools, logger: RRTAnalysis):
        """
        Extends RRT in the direction of the newly sampled node
        Input:
            rrt (RRT_tools): instance of RRT tree
        Output: 
            rrt after an extend operation 
        """
        #Find nearest neighbor node 
        nearest_neighbor_node = rrt.find_nearest_node_in_RRT_graph(new_q_sample)
        #Line Search to find how far to extend 
        feasible_path = rrt.calc_intermediate_qs_wo_collision(nearest_neighbor_node.value,new_q_sample)
        #Pick the configuration space point as far as possible in direction of sample 
        furthest_safe_q = feasible_path[-1]
        new_child_node = rrt.grow_rrt_tree(parent_node=nearest_neighbor_node, q_sample=furthest_safe_q)
        #add new node to dict for kd-tree
        rrt.rrt_tree.configurations_dict[new_child_node.value] = new_child_node
        logger.log_connection_length(nearest_neighbor_node, new_child_node)
        return rrt, new_child_node
    def rrt_connect_operation(self, rrt_A_child_node: TreeNode, rrt_B: RRT_tools):
        """
        Attempts to connect the latest child node of tree A to the nearet neighbor in tree B 
        Input:
            rrt_A:
            rrt_A_child_node,
            rrt_B: 
        Output: 
            (bool): True if the two trees can be connected
            nearest_neighbor_node (TreeNode): Child node of rrt_B that can be connected to rrt_A if success else None
        """ 
        #Find nearest neighbor in tree B
        nearest_neighbor_node = rrt_B.find_nearest_node_in_RRT_graph(rrt_A_child_node.value)
        #Line Search to find how far to extend 
        feasible_path = rrt_B.calc_intermediate_qs_wo_collision(nearest_neighbor_node.value,rrt_A_child_node.value)
        #Check how far the line search result got 
        furthest_safe_q = feasible_path[-1]
        if furthest_safe_q == rrt_A_child_node.value:
            rrt_B.grow_rrt_tree(parent_node=nearest_neighbor_node, q_sample=furthest_safe_q)
            return True, nearest_neighbor_node
        else:
            return False, None 
    def rrt_connect_planning(self, problem, logger, max_iterations=10e3, prob_sample_q_endpoints=0.05,):
        """
        Input:
            problem (IiwaProblem): instance of a utility class
            max_iterations: the maximum number of samples to be collected 
            prob_sample_q_goal: the probability of sampling q_goal

        Output:
            path (list): [q_start, ...., q_goal]. 
                        Note q's are configurations, not RRT nodes 
        """
        print("running RRT Connect")
        start_to_goal_rrt_tools = RRT_tools(self.iiwa_problem,"start")
        goal_to_start_rrt_tools = RRT_tools(self.iiwa_problem,"goal")
        q_goal = problem.goal
        q_start = problem.start
        RUN_RRT = True
        iters = 0
        rrt_connect_solution = []
        connecting_node = None 
        logger.register_RRT_Connect_algo(start_to_goal_rrt_tools,goal_to_start_rrt_tools)
        def backup_path(rrt_A,rrt_A_child_node,rrt_B,rrt_B_child_node):
            """
            Function to back up two paths on the two RRT-connect trees and report the combined path from start to goal 
            """

            rrt_A_solution = rrt_A.backup_path_from_node(rrt_A_child_node)
            rrt_B_solution = rrt_B.backup_path_from_node(rrt_B_child_node)
            rrt_B_solution.reverse()
            full_path = rrt_A_solution + rrt_B_solution
            return full_path
        while RUN_RRT and iters < max_iterations:
            
            if iters % 2 == 0:
                # Extend tree from start to goal
                #sample goal with prob p:
                rand_num = random.random()
                if rand_num < prob_sample_q_endpoints:
                    new_configuration_space_sample = q_goal
                else:
                #else sample random point in cspace
                    new_configuration_space_sample = start_to_goal_rrt_tools.sample_node_in_configuration_space()
                # Extend tree 
                start_to_goal_rrt_tools, newest_child_node = self.rrt_extend_operation(new_configuration_space_sample,start_to_goal_rrt_tools, logger)
                # Attempt to connect from the goal rooted tree 
                success, connecting_node = self.rrt_connect_operation(newest_child_node,
                                                                      goal_to_start_rrt_tools)
                if success:
                    RUN_RRT = False
                    rrt_connect_solution = backup_path(start_to_goal_rrt_tools,newest_child_node,goal_to_start_rrt_tools,connecting_node)

            else:
                # Extend tree from goal to start 
                #sample goal with prob p:
                rand_num = random.random()
                if rand_num < prob_sample_q_endpoints:
                    new_configuration_space_sample = q_start
                else:
                #else sample random point in cspace
                    new_configuration_space_sample = goal_to_start_rrt_tools.sample_node_in_configuration_space()
                # Extend tree 
                goal_to_start_rrt_tools, newest_child_node = self.rrt_extend_operation(new_configuration_space_sample,goal_to_start_rrt_tools, logger)
                #print(newest_child_node.value)
                # Attempt to connect from the goal rooted tree 
                success, connecting_node = self.rrt_connect_operation(newest_child_node,
                                                                      start_to_goal_rrt_tools)
    
                if success:
                    RUN_RRT = False
                    rrt_connect_solution = backup_path(start_to_goal_rrt_tools,connecting_node,goal_to_start_rrt_tools,newest_child_node)
            iters += 1
            logger.monitor_RRT_size()
        print("found path!")
        return rrt_connect_solution
    
    def rrt_planning(self,problem, logger, max_iterations=50e3, prob_sample_q_goal=.05):
        """
        Input:
            problem (IiwaProblem): instance of a utility class
            max_iterations: the maximum number of samples to be collected 
            prob_sample_q_goal: the probability of sampling q_goal

        Output:
            path (list): [q_start, ...., q_goal]. 
                        Note q's are configurations, not RRT nodes 
        """
        print("running RRT")
        rrt_tools = RRT_tools(self.iiwa_problem)
        q_goal = problem.goal
        q_start = problem.start
        RUN_RRT = True
        iters = 0
        rrt_solution = []
        logger = RRTAnalysis()
        logger.register_RRT_algo(rrt_tools)
        while RUN_RRT and iters < max_iterations:
            #sample goal with prob p:
            rand_num = random.random()
            if rand_num < prob_sample_q_goal:
                new_configuration_space_sample = q_goal
            else:
            #else sample random point in cspace
                new_configuration_space_sample = rrt_tools.sample_node_in_configuration_space()
            
            rrt_tools, new_child_node = self.rrt_extend_operation(new_configuration_space_sample,rrt_tools, logger)
            logger.monitor_RRT_size()
            #Check if the new node is within tolerance to goal 
            if rrt_tools.node_reaches_goal(node=new_child_node):
                print("Found Goal!")
                #We're done, now backup to get the path 
                rrt_solution = rrt_tools.backup_path_from_node(new_child_node)
                RUN_RRT = False
            iters += 1
        if len(rrt_solution) == 0:
            print("No solution was found within iteration limit")
        return rrt_solution
    def make_RRT_trajectory(self, rrt_solution: list, total_trajectory_time: int):
        """
        Constructs trajectory from list of configurations returned by the RRT algorithm 
        Input:
            rrt_solution: list of configuration vectors 
            total_trajectory_time: time to move between start and goal configuration
        Output:
            trajectory: (function(x)) function that returns the interpolated configuration vector from  0<=t<=total_trajectory_time 
        """
        x = np.linspace(0, total_trajectory_time,len(rrt_solution))
        y = np.array(rrt_solution)
        trajectory = scipy.interpolate.interp1d(x, y.T)
        return trajectory 
    def render_RRT_Solution(self,rrt_solution: list):
        """
        Animate RRT path 
        """
        animation_env = ManipulationStationSim(self.meshcat,True)
        animation_env.DrawStation(animation_env.q0,self.gripper_setpoint,self.left_door_angle,self.right_door_angle)
        time.sleep(3)
        if len(rrt_solution) == 0:
            raise ValueError("path to animate cannot be empty.")
        #start recording 
        print("Animating Trajectory....")
        diagram = animation_env.diagram
        diagram_context = animation_env.context_diagram
        t = 0 
        time_end = 10
        dt = 0.01
        config_space_trajectory = self.make_RRT_trajectory(rrt_solution,time_end)
        sim = Simulator(diagram, diagram_context)
        animation_env.station.GetInputPort("wsg_position").FixValue(animation_env.context_station,self.gripper_setpoint)
        animation_env.viz.StartRecording(set_transforms_while_recording = False)
        while t <= time_end:
            #update configuration 
            current_configuration = config_space_trajectory(t)
            animation_env.station.GetInputPort("iiwa_position").FixValue(animation_env.context_station,current_configuration)           
            t+=dt 
            sim.AdvanceTo(t)

        animation_env.viz.StopRecording()
        animation_env.viz.PublishRecording()
        #stop recording 
    
kuka_rrt = KukaRRTPlanner()
start_time = time.time()
logger = RRTAnalysis()
path_to_goal = kuka_rrt.rrt_connect_planning(kuka_rrt.iiwa_problem, logger)
end_time = time.time()
print(f"time taken: {end_time - start_time}")
kuka_rrt.render_RRT_Solution(path_to_goal)
plt.plot(logger.time_log,logger.start_to_goal_tree_size_log)
plt.show()
plt.plot(logger.time_log,logger.goal_to_start_tree_size_log)
plt.show()
