from rrt_tools import RRT_tools
from kuka_sim import *
import random 


class KukaRRTPlanner():
    def __init__(self) -> None:
        is_visualizing = True 
        self.R_WG = RotationMatrix(np.array([[0,1,0], [1,0,0], [0,0,-1]]).T)
        #Goal outside shelf 
        self.T_WG_goal = RigidTransform(p=np.array([4.69565839e-01, 2.95894043e-16, 0.65]), R=self.R_WG)
        #Goal inside shelf
        #self.T_WG_goal = RigidTransform(p=np.array([8.19565839e-01, 2.95894043e-16, 0.65]), R=self.R_WG)
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
    
    def rrt_planning(self,problem, max_iterations=50e3, prob_sample_q_goal=.05):
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
        while RUN_RRT and iters < max_iterations:
            #sample goal with prob p:
            rand_num = random.random()
            if rand_num < prob_sample_q_goal:
                new_configuration_space_sample = q_goal
            else:
            #else sample random point in cspace
                new_configuration_space_sample = rrt_tools.sample_node_in_configuration_space()
            #Find nearest neighbor node 
            nearest_neighbor_node = rrt_tools.find_nearest_node_in_RRT_graph(new_configuration_space_sample)
            #Line Search to find how far to extend 
            feasible_path = rrt_tools.calc_intermediate_qs_wo_collision(nearest_neighbor_node.value,new_configuration_space_sample)
            #Pick the configuration space point as far as possible in direction of sample 
            furthest_safe_q = feasible_path[-1]
            new_child_node = rrt_tools.grow_rrt_tree(parent_node=nearest_neighbor_node, q_sample=furthest_safe_q)
            #add new node to dict for kd-tree
            rrt_tools.rrt_tree.configurations_dict[new_child_node.value] = new_child_node
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
        animation_env.viz.StartRecording(set_transforms_while_recording = True)
        diagram = animation_env.diagram
        diagram_context = animation_env.context_diagram
        for kuka_q in rrt_solution:
            animation_env.DrawStation(kuka_q,self.gripper_setpoint,self.left_door_angle,self.right_door_angle)
        animation_env.viz.StopRecording()
        animation_env.viz.PublishRecording()
        #stop recording 
    
kuka_rrt = KukaRRTPlanner()
start_time = time.time()
path_to_goal = kuka_rrt.rrt_planning(kuka_rrt.iiwa_problem)
end_time = time.time()
print(f"time taken: {end_time - start_time}")
kuka_rrt.render_RRT_Solution(path_to_goal)