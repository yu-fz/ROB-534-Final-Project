import time
import numpy as np

from kuka_rrt_planner import KukaRRTPlanner, RRTAnalysis

TRIALS = 10

if __name__ == '__main__':
    rrt_cost = []
    postprocess_cost = []
    rrt_time = []
    postprocess_time = []
    for trial in range(TRIALS):
        kuka_rrt = KukaRRTPlanner()
        logger = RRTAnalysis()

        start_rrt = time.time()
        path_to_goal = kuka_rrt.rrt_connect_planning(kuka_rrt.iiwa_problem, logger)
        end_rrt = time.time()

        rrt_cost += [kuka_rrt.check_path_cost(path_to_goal)]
        rrt_time += [end_rrt - start_rrt]
        print(f"Trial {trial+1} unoptimized path cost: {rrt_cost[-1]:.3f}")
        print(f"Trial {trial+1} time spent for rrt connect: {rrt_time[-1]:.3f}s")

        kuka_rrt.render_RRT_Solution(path_to_goal)
        input()

        postprocess_start = time.time()
        better_path = kuka_rrt.post_process_rrt_path(path_to_goal)
        postprocess_end = time.time()

        postprocess_cost += [kuka_rrt.check_path_cost(better_path)]
        postprocess_time += [postprocess_end - postprocess_start]
        print(f"Trial {trial+1} postprocessed path cost: {postprocess_cost[-1]:.3f}")
        print(f"Trial {trial+1} time spent for postprocess: {postprocess_time[-1]:.3f}s")

        assert(np.array_equal(np.array(better_path[0]),kuka_rrt.q_start))
        assert(np.array_equal(np.array(better_path[-1]),kuka_rrt.q_goal))

        kuka_rrt.render_RRT_Solution(better_path)
        input()
# plt.plot(logger.time_log,logger.start_to_goal_tree_size_log)
# plt.show()
# plt.plot(logger.time_log,logger.goal_to_start_tree_size_log)
# plt.show()
