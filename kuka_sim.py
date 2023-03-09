import time
from random import random
from math import ceil, sqrt
import numpy as np
from manipulation import running_as_notebook, FindResource
from manipulation.exercises.trajectories.rrt_planner.robot import (
    ConfigurationSpace, Range)
from manipulation.exercises.trajectories.rrt_planner.rrt_planning import \
    Problem
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.scenarios import MakeManipulationStation
from pydrake.all import (DiagramBuilder, FindResourceOrThrow,
                         MeshcatVisualizer, MultibodyPlant, Parser,
                         RigidTransform, RollPitchYaw, RotationMatrix,
                         Simulator, SolutionResult, Solve, StartMeshcat, MeshcatVisualizerParams)
from pydrake.multibody import inverse_kinematics
from scipy.spatial import KDTree
from pydrake.geometry import (
    DrakeVisualizer,
    DrakeVisualizerParams,
    Role, Rgba
)
from pydrake.all import PiecewisePolynomial

class ManipulationStationSim:
    def __init__(self, meshcat, is_visualizing=False):
        builder = DiagramBuilder()
        self.station = builder.AddSystem(
            MakeManipulationStation(filename="./manipulation_station_with_cupboard.dmd.yaml",
                                    time_step=1e-3))
        self.plant = self.station.GetSubsystemByName("plant")
        self.scene_graph = self.station.GetSubsystemByName("scene_graph")
        self.is_visualizing = is_visualizing
        self.meshcat = meshcat

        # scene graph query output port.
        self.query_output_port = self.scene_graph.GetOutputPort("query")
        self._collision_params = MeshcatVisualizerParams(
            role=Role.kProximity,
            prefix="collision",
            default_color=Rgba(0.8, 0.5, 0.2, 0.3),
        )
        self._illustration_params = MeshcatVisualizerParams(
            role=Role.kIllustration,
            prefix="illustration",
            default_color=Rgba(0.9, 0.9, 0.9, 0.8),
        )
        # meshcat visualizer
        if is_visualizing:
            self.viz = MeshcatVisualizer.AddToBuilder(
                builder,
                self.station.GetOutputPort("query_object"),
                self.meshcat,self._illustration_params)
            MeshcatVisualizer.AddToBuilder(
                builder,
                self.station.GetOutputPort("query_object"),
                self.meshcat,self._collision_params)

        self.diagram = builder.Build()

        # contexts
        self.context_diagram = self.diagram.CreateDefaultContext()
        self.context_station = self.diagram.GetSubsystemContext(
            self.station, self.context_diagram)
        self.context_scene_graph = self.station.GetSubsystemContext(
            self.scene_graph, self.context_station)
        self.context_plant = self.station.GetMutableSubsystemContext(
            self.plant, self.context_station)
        # mark initial configuration
        self.q0 = self.plant.GetPositions(
            self.context_plant, self.plant.GetModelInstanceByName("iiwa"))
        if is_visualizing:
            self.DrawStation(self.q0, 0.1, -np.pi/2, np.pi/2)

    def SetStationConfiguration(self, q_iiwa, gripper_setpoint, left_door_angle,
                                right_door_angle):
        """
        :param q_iiwa: (7,) numpy array, joint angle of robots in radian.
        :param gripper_setpoint: float, gripper opening distance in meters.
        :param left_door_angle: float, left door hinge angle, \in [0, pi/2].
        :param right_door_angle: float, right door hinge angle, \in [0, pi/2].
        :return:
        """
        self.plant.SetPositions(self.context_plant,
                                self.plant.GetModelInstanceByName("iiwa"),
                                q_iiwa)
        self.plant.SetPositions(self.context_plant,
                                self.plant.GetModelInstanceByName("wsg"),
                                [-gripper_setpoint / 2, gripper_setpoint / 2])

        # cabinet doors
        if left_door_angle > 0:
            left_door_angle *= -1
        left_hinge_joint = self.plant.GetJointByName("left_door_hinge")
        left_hinge_joint.set_angle(context=self.context_plant,
                                   angle=left_door_angle)

        right_hinge_joint = self.plant.GetJointByName("right_door_hinge")
        right_hinge_joint.set_angle(context=self.context_plant,
                                    angle=right_door_angle)

    def DrawStation(self, q_iiwa, gripper_setpoint, q_door_left, q_door_right):
        if not self.is_visualizing:
            print("collision checker is not initialized with visualization.")
            return
        self.SetStationConfiguration(
            q_iiwa, gripper_setpoint, q_door_left, q_door_right)
        self.diagram.ForcedPublish(self.context_diagram)

    def ExistsCollision(self, q_iiwa, gripper_setpoint, q_door_left, q_door_right):

        self.SetStationConfiguration(
            q_iiwa, gripper_setpoint, q_door_left, q_door_right)
        query_object = self.query_output_port.Eval(self.context_scene_graph)
        collision_paris = query_object.ComputePointPairPenetration()

        return len(collision_paris) > 0

class IiwaProblem(Problem):
    def __init__(self,
                 q_start: np.array,
                 q_goal: np.array,
                 gripper_setpoint: float,
                 left_door_angle: float,
                 right_door_angle: float,
                 meshcat,
                 is_visualizing=False):
        self.gripper_setpoint = gripper_setpoint
        self.left_door_angle = left_door_angle
        self.right_door_angle = right_door_angle
        self.is_visualizing = is_visualizing

        self.collision_checker = ManipulationStationSim(meshcat,
            is_visualizing=is_visualizing)

        # Construct configuration space for IIWA.
        plant = self.collision_checker.plant
        nq = 7
        joint_limits = np.zeros((nq, 2))
        for i in range(nq):
            joint = plant.GetJointByName("iiwa_joint_%i" % (i + 1))
            joint_limits[i, 0] = joint.position_lower_limits()
            joint_limits[i, 1] = joint.position_upper_limits()

        range_list = []
        for joint_limit in joint_limits:
            range_list.append(Range(joint_limit[0], joint_limit[1]))

        def l2_distance(q: tuple):
            sum = 0
            for q_i in q:
                sum += q_i ** 2
            return np.sqrt(sum)

        max_steps = nq * [np.pi / 180 * 2]  # three degrees
        cspace_iiwa = ConfigurationSpace(range_list, l2_distance, max_steps)

        # Call base class constructor.
        Problem.__init__(
            self,
            x=10,  # not used.
            y=10,  # not used.
            robot=None,  # not used.
            obstacles=None,  # not used.
            start=tuple(q_start),
            goal=tuple(q_goal),
            cspace=cspace_iiwa)

    def collide(self, configuration):
        q = np.array(configuration)
        return self.collision_checker.ExistsCollision(
                    q, self.gripper_setpoint, self.left_door_angle, self.right_door_angle)

    def visualize_path(self, path):
        if path is not None:
            # show path in meshcat
            for q in path:
                q = np.array(q)
                self.collision_checker.DrawStation(
                    q, self.gripper_setpoint, self.left_door_angle,
                    self.right_door_angle)
                if running_as_notebook:
                    time.sleep(0.2)


class IKSolver(object):
    def __init__(self):
        ## setup controller plant
        plant_iiwa = MultibodyPlant(0.0)
        iiwa_file = FindResourceOrThrow(
        "drake/manipulation/models/iiwa_description/iiwa7/"
        "iiwa7_with_box_collision.sdf")
        iiwa = Parser(plant_iiwa).AddModelFromFile(iiwa_file)
        # Define frames
        world_frame = plant_iiwa.world_frame()
        L0 = plant_iiwa.GetFrameByName("iiwa_link_0")
        l7_frame = plant_iiwa.GetFrameByName("iiwa_link_7")
        plant_iiwa.WeldFrames(world_frame, L0)
        plant_iiwa.Finalize()
        plant_context = plant_iiwa.CreateDefaultContext()

        # gripper in link 7 frame
        X_L7G = RigidTransform(rpy=RollPitchYaw([np.pi/2, 0, np.pi/2]),
                                    p=[0,0,0.114])
        world_frame = plant_iiwa.world_frame()

        self.world_frame = world_frame
        self.l7_frame = l7_frame
        self.plant_iiwa = plant_iiwa
        self.plant_context = plant_context
        self.X_L7G = X_L7G

    def solve(self, X_WT, q_guess = None, theta_bound=0.01, position_bound=0.01):
        """
        plant: a mini plant only consists of iiwa arm with no gripper attached
        X_WT: transform of target frame in world frame
        q_guess: a guess on the joint state sol
        """
        plant = self.plant_iiwa
        l7_frame = self.l7_frame
        X_L7G = self.X_L7G
        world_frame = self.world_frame

        R_WT = X_WT.rotation()
        p_WT = X_WT.translation()

        if q_guess is None:
            q_guess = np.zeros(7)

        ik_instance = inverse_kinematics.InverseKinematics(plant)
        # align frame A to frame B
        ik_instance.AddOrientationConstraint(frameAbar=l7_frame,
                                        R_AbarA=X_L7G.rotation(),
                                        #   R_AbarA=RotationMatrix(), # for link 7
                                        frameBbar=world_frame,
                                        R_BbarB=R_WT,
                                        theta_bound=position_bound)
        # align point Q in frame B to the bounding box in frame A
        ik_instance.AddPositionConstraint(frameB=l7_frame,
                                        p_BQ=X_L7G.translation(),
                                        # p_BQ=[0,0,0], # for link 7
                                    frameA=world_frame,
                                    p_AQ_lower=p_WT-position_bound,
                                    p_AQ_upper=p_WT+position_bound)
        prog = ik_instance.prog()
        prog.SetInitialGuess(ik_instance.q(), q_guess)
        result = Solve(prog)
        if result.get_solution_result() != SolutionResult.kSolutionFound:
            return result.GetSolution(ik_instance.q()), False
        return result.GetSolution(ik_instance.q()), True

class TreeNode:
    def __init__(self, value, parent=None):
        self.value = value  # tuple of floats representing a configuration
        self.parent = parent  # another TreeNode
        self.children = []  # list of TreeNodes

class RRT:
    """
    RRT Tree.
    """
    def __init__(self, root: TreeNode, cspace: ConfigurationSpace):
        self.root = root  # root TreeNode
        self.cspace = cspace  # robot.ConfigurationSpace
        self.size = 1  # int length of path
        self.max_recursion = 1000  # int length of longest possible path
        self.configurations_dict = {self.root.value: self.root} # dict of all nodes added to RRT tree, starting with a single root node

    def add_configuration(self, parent_node, child_value):
        child_node = TreeNode(child_value, parent_node)
        parent_node.children.append(child_node)
        self.size += 1
        return child_node
    def kd_nearest(self, configuration):
        """
        Finds the nearest node by distance to configuration in the configuration space using a KD-tree.
        Args:
            configuration: tuple of floats representing a configuration of a
                robot
        Returns:
            closest: TreeNode. the closest node in the configuration space
                to configuration
            distance: float. distance from configuration to closest
        """
        assert self.cspace.valid_configuration(configuration)
        # Take all c-space vector keys in configurations dict and convert it to np array
        configuration_vectors = list(self.configurations_dict.keys())
        # convert list of configuration vectors to np array
        all_configurations = np.array(configuration_vectors)
        # Build KD-tree
        config_kd_tree = KDTree(all_configurations)
        distance, index = config_kd_tree.query(x=configuration,k=1) #query KD tree for nearest neighbor
        # find which node is the nearest neighbor by querying the configurations dictionary
        closest = self.configurations_dict[configuration_vectors[index]]
        return closest, distance

    # Brute force nearest, handles general distance functions
    def nearest(self, configuration):
        """
        Finds the nearest node by distance to configuration in the
             configuration space.

        Args:
            configuration: tuple of floats representing a configuration of a
                robot

        Returns:
            closest: TreeNode. the closest node in the configuration space
                to configuration
            distance: float. distance from configuration to closest
        """
        assert self.cspace.valid_configuration(configuration)
        def recur(node, depth=0):
            closest, distance = node, self.cspace.distance(node.value, configuration)
            if depth < self.max_recursion:
                for child in node.children:
                    (child_closest, child_distance) = recur(child, depth+1)
                    if child_distance < distance:
                        closest = child_closest
                        child_distance = child_distance
            return closest, distance
        return recur(self.root)[0]
