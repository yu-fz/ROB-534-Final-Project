from kuka_sim import RRT, TreeNode
import numpy as np
class RRT_tools:
    def __init__(self, problem):
        # rrt is a tree 
        self.rrt_tree = RRT(TreeNode(problem.start), problem.cspace)
        problem.rrts = [self.rrt_tree]
        self.problem = problem
        
    def find_nearest_node_in_RRT_graph(self, q_sample):
        #nearest_node = self.rrt_tree.nearest(q_sample)
        nearest_node,_ = self.rrt_tree.kd_nearest(q_sample) #much faster 
        return nearest_node
    
    def sample_node_in_configuration_space(self):
        q_sample = self.problem.cspace.sample()
        return q_sample
    
    def calc_intermediate_qs_wo_collision(self, q_start, q_end):
        '''create more samples by linear interpolation from q_start
        to q_end. Return all samples that are not in collision
        
        Example interpolated path: 
        q_start, qa, qb, (Obstacle), qc , q_end
        returns >>> q_start, qa, qb
        '''
        return self.problem.safe_path(q_start, q_end)
    
    def grow_rrt_tree(self, parent_node, q_sample):
        """
        add q_sample to the rrt tree as a child of the parent node
        returns the rrt tree node generated from q_sample
        """
        child_node = self.rrt_tree.add_configuration(parent_node, q_sample)
        return child_node
    
    def node_reaches_goal(self, node):

        l2_distance = np.linalg.norm(np.array(node.value)-np.array(self.problem.goal))
        #print(l2_distance)
        if l2_distance < 0.001:
            return True
        return False
    
    def backup_path_from_node(self, node):
        path = [node.value]
        while node.parent is not None:
            node = node.parent
            path.append(node.value)
        path.reverse()
        return path