# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
# Generates file with solutions to the training instances.                      #
# Needs to be run once before training.                                         #
# Usage: python 02_get_instance_solutions.py <type> -j <njobs> -n <ninstances>  #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #

import glob
import argparse
import json
import threading
import utilities
import numpy as np
import multiprocessing as mp

from tqdm import tqdm


def solve_instance(in_queue, out_queue, k_sols):
    """
    Worker loop: fetch an instance, run an episode and record samples.
    Parameters
    ----------
    in_queue : queue.Queue
        Input queue from which instances are received.
    out_queue : queue.Queue
        Output queue in which to put solutions.
    k_sols: int
        Number of solutions to save
    """
    while not in_queue.empty():
        instance, seed = in_queue.get()
        # Initialize SCIP model
        m = utilities.init_scip_model(instance, seed, 300)

        # Solve and retrieve solutions
        m.optimize()
        solutions = m.getSols()

        # number of primal bound improvements
        # before finding the optimal solution
        # -- m.getNBestSolsFound()

        # save solutions to individual files
        solutions = solutions[:k_sols]
        for i, solution in enumerate(solutions):
            m.writeSol(solution, f'{instance[:-3]}-{i + 1}.sol')

        info = {
            'obj_val': m.getObjVal(),
            'nnodes': m.getNNodes(),
            'time': m.getSolvingTime(),
        }  # return solving statistics
        out_queue.put({instance: info})

        m.freeProb()
        m.freeTransform()


def collect_solutions(instances, random, n_jobs, k_sols):
    """
    Runs branch-and-bound episodes on the given set of instances
    and collects the best k_solutions, which are saved to files.

    Parameters
    ----------
    instances : list
        Instances to process
    random: np.random.Generator
        Random number generator
    n_jobs : int
        Number of jobs for parallel sampling.
    k_sols : int
        Number of solutions to save to file
    """
    in_queue = mp.Queue()
    out_queue = mp.Queue()
    for instance in instances:
        in_queue.put([instance, random.integers(2**31)])
    print(f'{len(instances)} instances on queue.')

    workers = []
    for i in range(n_jobs):
        p = threading.Thread(
            target=solve_instance,
            args=(in_queue, out_queue, k_sols),
            daemon=True)
        workers.append(p)
        p.start()

    solutions = {}
    for _ in tqdm(range(len(instances))):
        answer = out_queue.get()
        solutions.update(answer)

    for p in workers:
        p.join()

    return solutions


if __name__ == '__main__':
    # read default config file
    with open('config.json', 'r') as f:
        config = json.load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=config['problems'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=utilities.valid_seed,
        default=0,
    )
    parser.add_argument(
        '-j', '--njobs',
        help='Number of parallel jobs.',
        type=int,
        default=1,
    )
    parser.add_argument(
        '-k', '--ksols',
        help='Number of solutions to save.',
        type=int,
        default=10
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    difficulty = config['difficulty'][args.problem]
    for instance_type in ['train', 'valid']:
        instance_dir = f'data/{args.problem}/instances/{instance_type}_{difficulty}'
        instances = glob.glob(instance_dir + '/*.lp')
        obj_values = collect_solutions(instances, rng, args.njobs, args.ksols)
        with open(instance_dir + "/instance_solutions.json", "w") as f:
            json.dump(obj_values, f)
