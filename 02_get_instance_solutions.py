# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
# Generates file with solutions to the training instances.                      #
# Needs to be run once before training.                                         #
# Usage: python 02_get_instance_solutions.py <type> -j <njobs> -n <ninstances>  #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #

import glob
import argparse
import json
import threading
import multiprocessing as mp

from pyscipopt import scip


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
    # i = 0
    while not in_queue.empty():
        instance = in_queue.get()
        # Initialize SCIP model
        m = scip.Model()
        m.setIntParam('display/verblevel', 0)
        m.readProblem(f'{instance}')
        m.optimize()

        # print number of primal bound improvements
        print(m.getNBestSolsFound())

        # And retrieve solutions
        solutions = m.getSols()

        # TODO: Remove instance file if not enough solutions
        if len(solutions) < k_sols: continue
        # os.rename(instance, new_filename)

        # return objective value and save solutions
        out_queue.put({instance: m.getObjVal()})
        solutions = solutions[:k_sols]
        for i, solution in enumerate(solutions):
            m.writeSol(solution, f'{instance[:-3]}-{i + 1}.sol')

        m.freeProb()
        m.freeTransform()

        # i += 1


def collect_solutions(instances, n_jobs, k_sols):
    """
    Runs branch-and-bound episodes on the given set of instances
    and collects the best k_solutions, which are saved to files.

    Parameters
    ----------
    instances : list
        Instances to process
    n_jobs : int
        Number of jobs for parallel sampling.
    k_sols : int
        Number of solutions to save to file
    """
    in_queue = mp.Queue()
    out_queue = mp.Queue()
    for instance in instances:
        in_queue.put(instance)
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
    for _ in range(len(instances)):
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

    difficulty = config['difficulty'][args.problem]
    for instance_type in ['train', 'valid']:
        instance_dir = f'data/{args.problem}/instances/{instance_type}_{difficulty}'
        instances = glob.glob(instance_dir + '/*.lp')
        obj_values = collect_solutions(instances, args.njobs, args.ksols)
        with open(instance_dir + "/instance_solutions.json", "w") as f:
            json.dump(obj_values, f)
