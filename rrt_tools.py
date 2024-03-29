from kuka_sim import RRT, TreeNode
import numpy as np

class RRT_tools:
    def __init__(self, problem, root_node: str="start"):
        # rrt is a tree
        assert root_node == "start" or root_node == "goal"
        if root_node == "start":
            self.rrt_tree = RRT(TreeNode(problem.start), problem.cspace)
        else:
            self.rrt_tree = RRT(TreeNode(problem.goal), problem.cspace)
        problem.rrts = [self.rrt_tree]
        self.problem = problem

    def find_nearest_node_in_RRT_graph(self, q_sample, k: int = 1):
        nearest_node,_ = self.rrt_tree.kd_nearest(q_sample, k=k) #much faster
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
        if l2_distance == 0:
            return True
        return False

    def backup_path_from_node(self, node):
        path = [node.value]
        while node.parent is not None:
            node = node.parent
            path.append(node.value)
        path.reverse()
        return path

    def rewire(self, neighborhood, new_parent_candidate):
        assert new_parent_candidate.value in self.rrt_tree.configurations_dict
        for n in neighborhood:
            assert n.value in self.rrt_tree.configurations_dict

        for node in neighborhood:
            #check connection feasibility
            furthest_safe_q = self.calc_intermediate_qs_wo_collision(node.value,new_parent_candidate.value)
            if furthest_safe_q[-1] == new_parent_candidate.value:
                new_cost = new_parent_candidate.cost + np.linalg.norm(np.array(new_parent_candidate.value) - node.value)
                if new_cost < node.cost:
                    # Remove this node from its old parent
                    self.rrt_tree.configurations_dict[node.parent.value].children.remove(node.value)

                    # Add this node as a child of its new parent
                    self.rrt_tree.configurations_dict[new_parent_candidate.value].children += [node.value]
                    self.rrt_tree.configurations_dict[node.value].parent = new_parent_candidate
                    self.rrt_tree.configurations_dict[node.value].cost = new_cost
        self.rrt_tree.update_tree_cost(new_parent_candidate)