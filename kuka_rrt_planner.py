from rrt_tools import RRT_tools
from kuka_sim import *


class KukaRRTPlanner():
    def __init__(self,do_viz: bool) -> None:
        self.meshcat = None
        if do_viz:
            self.meshcat = self._start_meshcat_vis()
            AddMeshcatTriad(self.meshcat, 'goal pose', X_PT=T_WG_goal, opacity=.5)
        self.env = ManipulationStationSim(self.meshcat,do_viz)
        self.q_start = self.env.q0
        R_WG = RotationMatrix(np.array([[0,1,0], [1,0,0], [0,0,-1]]).T)
        T_WG_goal = RigidTransform(p=np.array([4.69565839e-01, 2.95894043e-16, 0.65]), R=R_WG)
        ik_solver = IKSolver()
        self.q_goal, optimal = ik_solver.solve(T_WG_goal, q_guess=self.q_start)
        self.gripper_setpoint = 0.1
        self.door_angle = np.pi/2 - 0.001
        self.left_door_angle = -np.pi/6
        self.right_door_angle = np.pi/2

        self.iiwa_problem = IiwaProblem(
            q_start=self.q_start,
            q_goal=self.q_goal,
            gripper_setpoint=self.gripper_setpoint,
            left_door_angle=self.left_door_angle,
            right_door_angle=self.right_door_angle,
            meshcat = self.meshcat,
            is_visualizing=False)
        if do_viz:
            self.env.DrawStation(self.q_goal,self.gripper_setpoint,self.left_door_angle,self.right_door_angle)
    
    def _start_meshcat_vis(self):
        """
        Starts meshcat instance
        """
        # Start the visualizer.
        meshcat = StartMeshcat()
        print("loading meshcat...")
        time.sleep(2.5) #need to wait for meshcat server to initialize
        return meshcat 
    
    def rrt_planning(self,problem, max_iterations=1000, prob_sample_q_goal=.05):
        """
        Input: 
            problem (IiwaProblem): instance of a utility class
            max_iterations: the maximum number of samples to be collected 
            prob_sample_q_goal: the probability of sampling q_goal

        Output:
            path (list): [q_start, ...., q_goal]. 
                        Note q's are configurations, not RRT nodes 
        """
        rrt_tools = RRT_tools(self.iiwa_problem)
        q_goal = problem.goal
        q_start = problem.start

        RUN_RRT = True
        while RUN_RRT:
            #sample random point in cspace
            new_rrt_node = rrt_tools.sample_node_in_configuration_space()
            RUN_RRT = False
        return new_rrt_node
    
do_viz = False
kuka_rrt = KukaRRTPlanner(do_viz=do_viz)
print(kuka_rrt.rrt_planning(kuka_rrt.iiwa_problem))