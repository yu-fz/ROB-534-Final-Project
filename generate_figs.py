import ray
import time
import numpy as np

from kuka_rrt_planner import KukaRRTPlanner, RRTAnalysis


@ray.remote(num_cpus=1, max_retries=0)
def rrt_connect(*args, retry_failures: bool = False, postprocess: bool = False, rewire: bool = False, **kwargs):
    kuka_rrt = KukaRRTPlanner()
    logger = RRTAnalysis()

    failures = 0
    while True:
        try:
            start = time.time()
            path = kuka_rrt.rrt_connect_planning(kuka_rrt.iiwa_problem, logger, *args, rewire=rewire, **kwargs)

            if postprocess:
                path = kuka_rrt.post_process_rrt_path(path)

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
    futs = {}
    for algo in algos:
        futs[algo] = []
        for trial in range(trials):
            return_first = True if 'CaKC' not in algo else False
            while unused_cpus() == 0:
                time.sleep(1)
            print(f"Dispatching job for algo {algo}...")
            fut = rrt_connect.remote(max_iterations=max_iterations, return_first=return_first, retry_failures=True, postprocess='PP' in algo)
            futs[algo] += [fut]

    for algo in algos:
        for fut in futs[algo]:
            path, elapsed, cost, fails = ray.get(fut)
            algos[algo]['cost'] = cost
            algos[algo]['time'] = elapsed
            algos[algo]['path length'] = len(path)

            print(f"{algo} Trial {trial+1} path length: {algos[algo]['path length'][-1]}")
            print(f"{algo} Trial {trial+1} path cost: {algos[algo]['cost'][-1]:.3f}")
            print(f"{algo} Trial {trial+1} time: {algos[algo]['time'][-1]:.4f}s")

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


def generate_fig_1(trials: int = 5):
    import pathlib, pickle
    ray.init(num_cpus=8)
    print("Generating Figure 1...")
    futs = {}
    #for max_iterations in [1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 9e3, 1e4, 1.5e4, 2e4, 2.5e4, 3e4, 3.5e4]:
    for i in list(range(85)):
        max_iterations = int(3e3 + i * 1e3/2)

        futs[max_iterations] = []
        fname = f'iter-{max_iterations}-trials-{trials}'

        if pathlib.Path(fname).exists():
            print(f"{fname} cache exists, skipping max iters {max_iterations} and trials {trials}")
            continue

        for trial in range(trials):
            start = time.time()
            while unused_cpus() == 0:
                time.sleep(1)

            print(f"Dispatching job with max iters {max_iterations} trial {trial+1} of {trials}...")
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

        fname = f'iter-{max_iterations}-trials-{trials}'

        if pathlib.Path(fname).exists():
            print(f"Loading cache {fname}")
            with open(fname, 'rb') as f:
                data = pickle.load(f)
                times[max_iterations] = data['times']
                costs[max_iterations] = data['costs']
                lengths[max_iterations] = data['lengths']
                failures[max_iterations] = data['failures']
        else:
            print(f"Making cache {fname}")
            for fut in futs[max_iterations]:
                ret = ray.get(fut)

                path, elapsed, cost, fails = ret
                times[max_iterations] += [elapsed]
                costs[max_iterations] += [cost]
                lengths[max_iterations] += [len(path)]
                failures[max_iterations] += fails
            failures[max_iterations] = failures[max_iterations] / (len(futs[max_iterations]) + failures[max_iterations])
            with open(fname, 'wb') as f:
                pickle.dump({'times': times[max_iterations], 'costs': costs[max_iterations], 'lengths': lengths[max_iterations], 'failures': failures[max_iterations]}, f)
        print(f"max iters {max_iterations} had failure rate {failures[max_iterations]*100:.1f}%, cost {np.mean(costs[max_iterations]):.2f}, elapsed {np.mean(times[max_iterations]):.2f}")

    # Generate a table
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

    # Generate a figure
    import matplotlib.pyplot as plt
    xs = sorted(list(costs.keys()))
    #fig, ((ax1, ax2, ax3)) = plt.subplots(3, 1, figsize=(4,8))
    fig, ((ax1)) = plt.subplots(1, 1, figsize=(4,4))
    data = [
        (ax1, np.array([np.mean(costs[k]) for k in xs]), np.array([np.std(costs[k]) for k in xs]), 'green', 'Mean path cost'),
        #(ax1, np.array([np.mean(failures[k]) for k in xs]), np.array([np.std(failures[k]) for k in xs]), 'green', 'Mean failure rate'),
        #(ax1, np.array([np.mean(times[k]) for k in xs]), np.array([np.std(times[k]) for k in xs]), 'green', 'Mean elapsed time'),
        #(ax2, [np.mean(times[k]) for k in xs], 'blue', 'Mean elapsed time'),
        #(ax3, [failures[k] for k in xs], 'red', 'Failure rate'),
    ]
    plt.title(f'RRT-CaKC over {trials} trials')
    for ax, ys, sd, color, desc in data:
        ax.plot(xs, ys, alpha=0.8, color=color)
        y_m = ys - sd
        y_p = ys + sd
        ax.fill_between(xs, y_m, y_p, alpha=0.4)
        #ax.set_title(f'{desc} of RRT-CaKC as function of max iterations')
        ax.set_ylim(bottom=0, top=max(y_p)*1.1)
        ax.set_xlabel('Maximum # of iterations')
        ax.set_ylabel(f'{desc}')
    fig.tight_layout()
    #plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.show()
    print("Showed plot")
    #plt.imsave('/tmp/fig2.png', )


if __name__ == '__main__':
    ray.init(num_cpus=1)
    fut = rrt_connect.remote(return_first=False, retry_failures=False, postprocess=False, rewire=True)
    ray.get(fut)
    #generate_fig_1()
    #generate_table_1()
    #generate_table_2()
