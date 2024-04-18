# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
# Generates file with solutions to the training instances.                      #
# Needs to be run once before training.                                         #
# Usage: python 02_get_instance_solutions.py <type> -j <njobs> -n <ninstances>  #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #

import glob
import json
import argparse
import utilities
import numpy as np
import pyscipopt as scip
import multiprocessing as mp

from tqdm import trange


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
        m = scip.Model()
        m.hideOutput()
        m.readProblem(instance)

        # 1: CPU user seconds, 2: wall clock time
        m.setIntParam('timing/clocktype', 1)
        m.setRealParam('limits/time', 300)
        utilities.init_scip_params(m, seed)

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

        # return optimal objective value
        out_queue.put({instance: m.getObjVal()})

        m.freeProb()


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
        p = mp.Process(
            target=solve_instance,
            args=(in_queue, out_queue, k_sols),
            daemon=True)
        workers.append(p)
        p.start()

    solutions = {}
    for _ in trange(len(instances)):
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
        default=config['seed'],
        type=utilities.valid_seed,
    )
    parser.add_argument(
        '-j', '--njobs',
        help='Number of parallel jobs.',
        default=1,
        type=int,
    )
    parser.add_argument(
        '-k', '--ksols',
        help='Number of solutions to save.',
        default=config['k'],
        type=int,
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    difficulty = config['difficulty'][args.problem]
    for instance_type in ['train', 'valid']:
        instance_dir = f'data/{args.problem}/instances/{instance_type}_{difficulty}'
        instances = glob.glob(instance_dir + '/*.lp')

        obj_values = collect_solutions(instances, rng, args.njobs, args.ksols)
        with open(instance_dir + f"/instance_solutions.json", "w") as f:
            json.dump(obj_values, f)
