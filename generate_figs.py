import time
import numpy as np

from kuka_rrt_planner import KukaRRTPlanner, RRTAnalysis

TRIALS = 15

if __name__ == '__main__':
    rrt_cost = []
    postprocess_cost = []
    rrt_time = []
    postprocess_time = []
    def _make_algo_dict(name):
        return {name: {'time': [], 'cost': [], 'path length': []}}
    algos = {**_make_algo_dict('RRT-C'),
             **_make_algo_dict('RRT-C + PP'),
             **_make_algo_dict('RRT-CaKC'),
             **_make_algo_dict('RRT-CaKC + PP')}
    for algo in algos:
        for trial in range(TRIALS):
            kuka_rrt = KukaRRTPlanner()
            logger = RRTAnalysis()

            return_first = True if 'CaKC' not in algo else False

            start = time.time()
            path_to_goal = kuka_rrt.rrt_connect_planning(kuka_rrt.iiwa_problem,
                                                         logger,
                                                         return_first=return_first)

            if 'PP' in algo:
                path_to_goal = kuka_rrt.post_process_rrt_path(path_to_goal)
            end = time.time()

            algos[algo]['cost'] += [kuka_rrt.check_path_cost(path_to_goal)]
            algos[algo]['time'] += [end - start]
            algos[algo]['path length'] += [len(path_to_goal)]

            print(f"{algo} Trial {trial+1} path length: {algos[algo]['path length'][-1]}")
            print(f"{algo} Trial {trial+1} path cost: {algos[algo]['cost'][-1]:.3f}")
            print(f"{algo} Trial {trial+1} time: {algos[algo]['time'][-1]:.4f}s")

            assert np.array_equal(np.array(path_to_goal[0]),kuka_rrt.q_start)
            assert np.array_equal(np.array(path_to_goal[-1]),kuka_rrt.q_goal)

            #kuka_rrt.render_RRT_Solution(better_path)
            #input()

    latex_str = "\\begin{table}[!h]\n\\begin{center}\n\\begin{tabular}{l|l|l|l}\nAlgorithm & Cost & Path Length & Time \\\\\n"
    for algo in algos:
        cost = np.mean(algos[algo]['cost'])
        length = np.mean(algos[algo]['path length'])
        elapsed = np.mean(algos[algo]['time'])
        latex_str += f"{algo} & {cost:.3f} & {length:.1f} & {elapsed:.1f}s\\\\\n"
    latex_str += f"\\end{{tabular}}\n\\caption{{Comparison of cost and time performance for each algorithm we tested. Experiments were run over {TRIALS} trials.}}\n\\end{{center}}\\end{{table}}"
    print(latex_str)
    print()

# plt.plot(logger.time_log,logger.start_to_goal_tree_size_log)
# plt.show()
# plt.plot(logger.time_log,logger.goal_to_start_tree_size_log)
# plt.show()
