# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
# Evaluate all GCNN models (il, mdp, tmdp+DFS, tmdp+ObjLim) and SCIP's default  #
# rule, on 2 benchmarks (test and transfer). Each instance-model pair is solved #
# with 5 different seeds. Output is written into a csv file.                    #
# Usage: python 05_evaluate.py <type> -g <cudaId>                               #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #

import os
import csv
import json
import time
import glob
import queue
import argparse
import utilities
import model as ml
import numpy as np
import torch as th
import pyscipopt as scip
import multiprocessing as mp

from tqdm import trange
from nodesels.nodesel_policy import NodeselPolicy
from scipy.stats import gmean


class NodeselBFS(scip.Nodesel):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "BFS"

    def nodeselect(self):
        selnode = self.model.getBestboundNode()
        return {'selnode': selnode}


class NodeselRandom(NodeselPolicy):
    def __init__(self):
        super().__init__(name="Random")
        self.random = np.random.default_rng(args.seed)

    def nodecomp(self, node1, node2):
        return -1 if self.random.random() < 0.5 else 1


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
        num_nodes = []
        solvetime = []
        instance, base_seed = in_queue.get()
        instance_name = os.path.basename(instance)
        for seed in range(base_seed, base_seed + 5):
            th.manual_seed(seed)

            # Initialize SCIP model
            m = scip.Model()
            m.hideOutput()
            m.readProblem(instance)

            # 1: CPU user seconds, 2: wall clock time
            m.setIntParam('timing/clocktype', 1)
            m.setRealParam('limits/time', 90)
            utilities.init_scip_params(m, seed, static)
            m.setRealParam('limits/objectivestop', instance['sol'])

            if nodesel is not None:
                m.includeNodesel(nodesel=nodesel,
                                 name=f"nodesel_{nodesel}",
                                 desc="nodesel to be evaluated",
                                 stdpriority=300000,
                                 memsavepriority=300000)

            # Solve and retrieve solutions
            wall_time = time.perf_counter()
            proc_time = time.process_time()

            m.optimize()

            wall_time = time.perf_counter() - wall_time
            proc_time = time.process_time() - proc_time

            num_nodes.append(m.getNNodes())
            solvetime.append(m.getSolvingTime())

            out_queue.put({
                'type': "row",
                'seed': seed,
                'instance': instance_name,
                'nnodes': m.getNNodes(),
                'nlps': m.getNLPs(),
                'gap': m.getGap(),
                'nsols': m.getNBestSolsFound(),
                'stime': m.getSolvingTime(),
                'walltime': wall_time,
                'proctime': proc_time,
            })

            m.freeProb()

        out_queue.put({
            'type': "mean",
            'instance': instance_name,
            'nnodes': np.mean(num_nodes),
            'solve_time': np.mean(solvetime),
        })


def collect_evaluation(instances, seed, n_jobs, nodesel, static, result_file):
    """
    Runs branch-and-bound episodes on the given set of instances
    with the provided node selector and settings.

    Parameters
    ----------
    instances : list
        Instances to process
    n_jobs : int
        Number of jobs for parallel sampling.
    nodesel : object
        Nodesel for which to evaluate
    """
    in_queue = mp.Queue()
    out_queue = mp.Queue()
    for instance in instances:
        in_queue.put([instance, seed])

    workers = []
    for i in range(n_jobs):
        p = mp.Process(
            target=evaluate,
            args=(in_queue, out_queue, nodesel, static),
            daemon=True)
        workers.append(p)
        p.start()

    nnodes = []
    stimes = []
    with open(result_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for _ in trange(6 * len(instances)):
            try:
                answer = out_queue.get(timeout=100)
            # if no response is given in time_limit seconds,
            # the solver has crashed and the worker is dead:
            # start a new worker to pick up the pieces.
            except queue.Empty:
                print("starting new worker")
                p = mp.Process(
                    target=evaluate,
                    args=(in_queue, out_queue, nodesel, static),
                    daemon=True)
                workers.append(p)
                p.start()
                continue
            if answer['type'] == "row":
                del answer['type']
                writer.writerow(answer)
                csvfile.flush()
            else:  # answer['type'] == "mean"
                nnodes.append(answer['nnodes'])
                stimes.append(answer['solve_time'])

    for p in workers:
        p.join()

    return {'nnodes_g': gmean(nnodes),
            'nnodes_std': np.std(nnodes),
            'stimes_g': gmean(stimes),
            'stimes_std': np.std(stimes),
            }


if __name__ == "__main__":
    # read default config file
    with open('config.json') as f:
        config = json.load(f)

    # read command-line arguments
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

    ### PYTORCH SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = 'cpu'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = f"cuda:0"

    # Default: BestEstimate, BFS, Random, Untrained Policy
    model = ml.MLPPolicy().to(device)
    nodesels = [None, NodeselBFS(), NodeselRandom(),
                NodeselPolicy(model, device, "policy")]
    nodesels = []

    # Learned models
    for model_id in ["il", "rl_mdp_"]:
        model_path = f'actor/{args.problem}/{model_id}.pkl'
        if os.path.exists(model_path):
            model = ml.MLPPolicy().to(device)
            model.load_state_dict(th.load(model_path))
            nodesel = NodeselPolicy(model, device, model_id)
            nodesels.append(nodesel)

    print(f"problem: {args.problem}")
    print(f"gpu: {args.gpu}")

    fieldnames = [
        'seed',
        'instance',
        'nnodes',
        'nlps',
        'gap',
        'nsols',
        'stime',
        'walltime',
        'proctime',
    ]

    transfer_difficulty = {
        'indset': "1000_4",
        'gisp': "80_0.5",
        'mkp': "100_8",
        'cflp': "60_35_5",
        'fcmcnf': "30_45_100",
        'setcover': "500_1000_0.05",
        'cauctions': "200_1000"
    }[args.problem]
    results = {}
    for instance_type in ["test", "transfer"]:
        instance_dir = f'data/{args.problem}/instances'
        difficulty = transfer_difficulty if instance_type == "transfer" else config['difficulty'][args.problem]
        instances = [str(file) for file in glob.glob(instance_dir + f'/{instance_type}_{difficulty}/*.lp')]

        # with open(instance_dir + f'/obj_values.json') as f:
        #     opt_sols = json.load(f)
        #
        # sign = 1 if args.problem in ["cflp"] else -1
        # valid_batch = [{'path': instance, 'seed': seed, 'sol': sign * opt_sols[instance]}
        #                for instance in instances for seed in range(config['num_seeds'])]

        timestamp = time.strftime('%Y-%m-%d--%H.%M.%S')
        experiment_dir = f'experiments/{args.problem}/05_evaluate'
        running_dir = experiment_dir + f'/{args.seed}_{timestamp}'
        os.makedirs(running_dir, exist_ok=True)

        for nodesel in nodesels:
            for static in [True, False]:
                env = "static" if static else "active"
                experiment_id = f"{instance_type}_{env}_{nodesel}"
                utilities.log(f"Starting experiment {experiment_id}")
                result_file = os.path.join(running_dir, f'{experiment_id}_results.csv')
                stats = collect_evaluation(instances, args.seed, args.njobs, nodesel, static, result_file)
                results[experiment_id] = stats
    print(results)
