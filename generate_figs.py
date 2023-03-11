import ray
import time
import numpy as np

from kuka_rrt_planner import KukaRRTPlanner, RRTAnalysis


@ray.remote
def rrt_connect(*args, retry_failures: bool = False, **kwargs):
    kuka_rrt = KukaRRTPlanner()
    logger = RRTAnalysis()

    failures = 0
    while True:
        try:
            start = time.time()
            path = kuka_rrt.rrt_connect_planning(kuka_rrt.iiwa_problem, logger, *args, **kwargs)
            end = time.time()
            return path, end - start, kuka_rrt.check_path_cost(path), failures
        except ValueError:
            if not retry_failures:
                raise ValueError
            failures += 1
            print(f"Caught a failure, retrying...")


def unused_cpus():
    assert ray.is_initialized()
    resources = ray.available_resources()
    return int(resources['CPU'] if 'CPU' in resources else 0)


def generate_table_1(trials: int = 15):
    print("Generating table 1...")
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
            while True:
                try:
                    path_to_goal = kuka_rrt.rrt_connect_planning(kuka_rrt.iiwa_problem,
                                                                 logger,
                                                                 return_first=return_first)
                    break
                except ValueError:
                    print(f"RRT failed to solve, trying again...")

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

def generate_table_2(trials: int = 10):
    print("Generating table 2...")
    failures = {}
    for i in range(10):
        max_iterations = (i+1) * 1e3
        failures[max_iterations] = 0
        for trial in range(trials):
            start = time.time()
            kuka_rrt = KukaRRTPlanner()
            logger = RRTAnalysis()
            try:
                start = time.time()
                path_to_goal = kuka_rrt.rrt_connect_planning(kuka_rrt.iiwa_problem,
                                                             logger,
                                                             max_iterations=max_iterations,
                                                             return_first=True)
                #assert path_to_goal is not None
                end = time.time()
                break
            except ValueError:
                failures[max_iterations] += 1
                print(f"Had failure, max iterations {max_iterations} has {failures[max_iterations]} fails for a rate of {failures[max_iterations] / trials}")
        failures[max_iterations] /= trials
        print(f"{max_iterations} has failure rate {failures[max_iterations]}")
    latex_str = "\\begin{table}[!h]\n\\begin{center}\n\\begin{tabular}{|l|l}\nMax Iterations of RRT-CaKC & Failure Rate \\\\\n"
    for max_iters in failures:
        failure_rate = failures[max_iters]
        latex_str += f"{int(max_iters):n} & {failure_rate * 100:.1f}%\\\\\n"
    latex_str += f"\\end{{tabular}}\n\\caption{{Failure rate of RRT-Connect as a function of the maximum number of iterations}}\n\\end{{center}}\\end{{table}}"
    print(latex_str)
    print()


def generate_fig_1(trials: int = 20):
    ray.init(num_cpus=8)
    print("Generating Figure 1...")
    futs = {}
    for max_iterations in [4e3, 5e3, 6e3, 7e3, 8e3, 9e3, 1e4]:#, 2e4, 3e4, 4e4, 5e4]:
        futs[max_iterations] = []
        for trial in range(trials):
            start = time.time()
            while unused_cpus() == 0:
                time.sleep(1)

            print(f"Dispatching job with max iters {max_iterations}...")
            fut = rrt_connect.remote(max_iterations=max_iterations, return_first=False, retry_failures=True)
            futs[max_iterations] += [fut]

    lengths = {}
    costs = {}
    times = {}
    failures = {}
    for max_iterations in futs:
        lengths[max_iterations] = []
        costs[max_iterations] = []
        times[max_iterations] = []
        failures[max_iterations] = 0
        for fut in futs[max_iterations]:
            ret = ray.get(fut)

            path, elapsed, cost, fails = ret
            times[max_iterations] += [elapsed]
            costs[max_iterations] += [cost]
            lengths[max_iterations] += [len(path)]
            failures[max_iterations] += fails
        failures[max_iterations] = failures[max_iterations] / (len(futs[max_iterations]) + failures[max_iterations])
        print(f"max iters {max_iterations} had failure rate {failures[max_iterations]*100:.1f}%, cost {np.mean(costs[max_iterations]):.2f}, elapsed {np.mean(times[max_iterations]):.2f}")

    latex_str = "\\begin{table}[!h]\n\\begin{center}\n\\begin{tabular}{l|l|l|l}\nMax Iterations of RRT-CaKC & Cost & Path Length & Time \\\\\n"
    for max_iters in costs:
        cost = np.mean(costs[max_iters])
        length = np.mean(lengths[max_iters])
        elapsed = np.mean(times[max_iters])
        fails = failures[max_iters]
        latex_str += f"{int(max_iters):n} & {cost:.3f} & {length:.1f} & {elapsed:.1f}s & {fails*100:.1f}%%\\\\\n"
    latex_str += f"\\end{{tabular}}\n\\caption{{Comparison of best cost and time performance for various maximum iteration limits for RRT-CaKC. Experiments were run over {trials} trials.}}\n\\end{{center}}\\end{{table}}"
    print(latex_str)
    print()

    xs = sorted(list(costs.keys()))
    ys = [np.mean(costs[k]) for k in xs]
    import matplotlib.pyplot as plt
    plt.plot(xs, ys, alpha=0.8, color='green')
    plt.title('Mean path cost of RRT-CaKC as function of max iterations')
    plt.xlabel('Maximum # of iterations')
    plt.ylabel(f'Mean path cost over {trials} trials')
    plt.show()
    #plt.imsave('/tmp/fig2.png', )


if __name__ == '__main__':
    generate_fig_1()
    #generate_table_1()
    #generate_table_2()
