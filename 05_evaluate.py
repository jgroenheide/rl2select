# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
# Evaluate all GCNN models (il, mdp, tmdp+DFS, tmdp+ObjLim) and SCIP's default  #
# rule, on 2 benchmarks (test and transfer). Each instance-model pair is solved #
# with 5 different seeds. Output is written into a csv file.                    #
# Usage:                                                                        #
# python 05_evaluate.py <type> -g <cudaId>                                         #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
import glob
import os
import csv
import json
import time
import argparse
import utilities
import model as ml
import numpy as np
import torch as th
import pyscipopt as scip
import multiprocessing as mp

from tqdm import trange
from nodesels import nodesel_policy


class NodeselBFS(scip.Nodesel):
    def __init__(self):
        super().__init__()

    def nodeselect(self):
        return {"selnode": self.model.getBestboundNode()}

    def __str__(self):
        return "BFS"


def evaluate(in_queue, out_queue, nodesel, static):
    """
    Worker loop: fetch an instance, run an episode and record samples.
    Parameters
    ----------
    in_queue : queue.Queue
        Input queue from which instances are received.
    out_queue : queue.Queue
        Output queue in which to put solutions.
    """
    while not in_queue.empty():
        instance, seed = in_queue.get()
        th.manual_seed(seed)

        # Initialize SCIP model
        m = scip.Model()
        m.hideOutput()
        m.readProblem(instance)

        # 1: CPU user seconds, 2: wall clock time
        m.setIntParam('timing/clocktype', 1)
        m.setRealParam('limits/time', 3600)
        utilities.init_scip_params(m, seed, static)

        if nodesel is not None:
            m.includeNodesel(nodesel=nodesel,
                             name="generic name",
                             desc='BFS node selector',
                             stdpriority=300000,
                             memsavepriority=300000)

        # Solve and retrieve solutions
        walltime = time.perf_counter()
        proctime = time.process_time()

        m.optimize()

        walltime = time.perf_counter() - walltime
        proctime = time.process_time() - proctime

        # number of primal bound improvements
        # before finding the optimal solution
        # -- m.getNBestSolsFound()

        stime = m.getSolvingTime()
        nnodes = m.getNNodes()
        nsols = m.getNBestSolsFound()
        nlps = m.getNLPs()
        gap = m.getGap()
        status = m.getStatus()

        out_queue.put({
            'instance': instance,
            'seed': seed,
            'nnodes': nnodes,
            'nsols': nsols,
            'nlps': nlps,
            'gap': gap,
            'status': status,
            'stime': stime,
            'walltime': walltime,
            'proctime': proctime,
        })

        m.freeProb()


def collect_evaluation(instances, random, n_jobs, nodesel, static, result_file):
    """
    Runs branch-and-bound episodes on the given set of instances
    with the provided node selector and settings.

    Parameters
    ----------
    instances : list
        Instances to process
    random: np.random.Generator
        Random number generator
    n_jobs : int
        Number of jobs for parallel sampling.
    nodesel : object
        Nodesel for which to evaluate
    """
    in_queue = mp.Queue()
    out_queue = mp.Queue()
    for instance in instances:
        in_queue.put([instance, random.integers(2 ** 31)])
    print(f'{len(instances)} instances on queue.')

    workers = []
    for i in range(n_jobs):
        p = mp.Process(
            target=evaluate,
            args=(in_queue, out_queue, nodesel, static),
            daemon=True)
        workers.append(p)
        p.start()

    with open(result_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for _ in trange(len(instances)):
            answer = out_queue.get()
            writer.writerow(answer)
            csvfile.flush()

    for p in workers:
        p.join()


if __name__ == "__main__":
    # read default config file
    with open("config.json", 'r') as f:
        config = json.load(f)

    # read command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=config['problems'],
    )
    parser.add_argument(
        'instance_type',
        help='Type of instances to sample',
        choices=['test', 'transfer'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        default=config['seed'],
        type=int,
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        default=config['gpu'],
        type=int,
    )
    parser.add_argument(
        '-j', '--njobs',
        help='Number of parallel jobs.',
        default=1,
        type=int,
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    ### PYTORCH SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = 'cpu'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = f"cuda:0"

    # Default: BestEstimate, BFS
    policies = [None, NodeselBFS()]

    # Learned models
    for mode in []:
        model = ml.MLPPolicy().to(device)
        model.load_state_dict(th.load(f'actor/{args.problem}/{mode}.pkl'))
        nodesel = nodesel_policy.NodeselPolicy(model, device, NodeselBFS())
        policies.append(nodesel)

    print(f"problem: {args.problem}")
    print(f"type: {args.instance_type}")
    print(f"gpu: {args.gpu}")

    fieldnames = [
        'instance',
        'seed',
        'nnodes',
        'nsols',
        'nlps',
        'gap',
        'status',
        'stime',
        'walltime',
        'proctime',
    ]

    transfer_difficulty = {
        "indset": "1000_4",
        "gisp": "80_0.5",
        "cflp": "60_35_5",
        "fcmcnf": "30_45_100",
        "setcover": "500_1000_0.05",
        "mknapsack": "100_12",
        "cauctions": "200_1000"
    }[args.problem]
    difficulty = transfer_difficulty if args.instance_type == "transfer" else config['difficulty'][args.problem]
    instance_dir = f"data/{args.problem}/instances/{args.instance_type}_{difficulty}"
    instances = glob.glob(instance_dir + '/*.lp')

    timestamp = time.strftime('%Y-%m-%d--%H.%M.%S')
    running_dir = f'experiments/{args.problem}_{difficulty}/{args.seed}_{timestamp}'
    os.makedirs(running_dir)
    for nodesel in policies:
        for static in [True, False]:
            static_ = "static_" if static else ""
            result_file = os.path.join(running_dir, f"{static_}{nodesel}_results.csv")
            collect_evaluation(instances, rng, args.njobs, nodesel, static, result_file)
