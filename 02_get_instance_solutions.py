# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
# Generates file with solutions to the training instances.                      #
# Needs to be run once before training.                                         #
# Usage: python 02_get_instance_solutions.py <type> -j <njobs> -n <ninstances>  #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #

import os
import glob
import json
import shutil
import argparse
import importlib
import utilities

import multiprocessing as mp
import networkx as nx
import numpy as np
import pyscipopt as scip

from tqdm import trange

gen = importlib.import_module('01_generate_instances')


def solve_instance(in_queue, out_queue, k_sols):
    """
    Worker loop: fetch an instance, run an episode and record samples.
    Parameters
    ----------
    in_queue : mp.JoinableQueue
        Input queue from which instances are received.
    out_queue : mp.SimpleQueue
        Output queue in which to put solutions.
    k_sols: int
        Number of solutions to save
    """
    while True:
        instance, seed = in_queue.get()

        # Initialize SCIP model
        m = scip.Model()
        m.hideOutput()
        m.readProblem(instance)

        # 1: CPU user seconds, 2: wall clock time
        m.setIntParam('timing/clocktype', 1)
        m.setRealParam('limits/time', 30)

        m.optimize()

        # Statistics to help tune new problems
        print(f"Status: {m.getStatus()}")
        print(f"NNodes: {m.getNNodes()}")
        print(f"NSols: {m.getNBestSolsFound()}")
        print(f"MaxDepth: {m.getMaxDepth()}")

        if m.getStatus() == "optimal" and 100 < m.getNNodes() < 1000:
            # retrieve and save solutions to individual files
            solutions = m.getSols()[:k_sols]
            for i, sol in enumerate(solutions):
                filename = f'{instance[:-3]}-{i + 1}.sol'
                m.writeSol(sol, filename)
            out_queue.put({'filename': instance,
                           'num_sols': len(solutions),
                           'opt_sol': m.getObjVal(),
                           'nnodes': m.getNNodes()})

        m.freeProb()
        in_queue.task_done()


def generate_instances(orders_queue, problem, random, transfer=False):
    out_dir = f'data/{problem}/instances'
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed=0)
    if problem == "indset":
        affinity = 4
        n_nodes = 1000 if transfer else 500
        tmp_dir = out_dir + f'/tmp_{n_nodes}_{affinity}'
        os.makedirs(tmp_dir, exist_ok=True)

        episode = 1
        while True:
            filename = tmp_dir + f'/instance_{episode}.lp'
            graph = gen.Graph.barabasi_albert(n_nodes, affinity, rng)
            gen.generate_indset(graph, filename)
            # blocks the process until a slot in the queue is available
            orders_queue.put([filename, random.integers(2 ** 31)])
            episode += 1

    elif problem == "gisp":
        edge_prob = 0.6
        drop_rate = 0.5
        n_nodes = 80 if transfer else 60
        tmp_dir = out_dir + f'/tmp_{n_nodes}_{drop_rate}'
        os.makedirs(tmp_dir, exist_ok=True)

        episode = 1
        while True:
            filename = tmp_dir + f'/instance_{episode}.lp'
            graph = gen.Graph.erdos_renyi(n_nodes, edge_prob, rng)
            gen.generate_general_indset(graph, filename, drop_rate, rng)
            # blocks the process until a slot in the queue is available
            orders_queue.put([filename, random.integers(2 ** 31)])
            episode += 1

    elif problem == "mkp":
        n_items = 100
        n_knapsacks = 8 if transfer else 4
        tmp_dir = out_dir + f'/tmp_{n_items}_{n_knapsacks}'
        os.makedirs(tmp_dir, exist_ok=True)

        episode = 1
        while True:
            filename = tmp_dir + f'/instance_{episode}.lp'
            gen.generate_mknapsack(n_items, n_knapsacks, filename, rng)
            # blocks the process until a slot in the queue is available
            orders_queue.put([filename, random.integers(2 ** 31)])
            episode += 1

    elif problem == "cflp":
        n_facilities = 25  # original: 35
        ratio = 3  # original: 5
        n_customers = 60 if transfer else 25  # original: 35
        tmp_dir = out_dir + f'/tmp_{n_customers}_{n_facilities}_{ratio}'
        os.makedirs(tmp_dir, exist_ok=True)

        episode = 1
        while True:
            filename = tmp_dir + f'/instance_{episode}.lp'
            gen.generate_capacitated_facility_location(n_customers, n_facilities, ratio, filename, rng)
            # blocks the process until a slot in the queue is available
            orders_queue.put([filename, random.integers(2 ** 31)])
            episode += 1

    elif problem == "fcmcnf":
        edge_prob = 0.33
        n_nodes = 20 if transfer else 15
        n_commodities = 30 if transfer else 22
        tmp_dir = out_dir + f'/tmp_{n_nodes}_{n_commodities}_{edge_prob}'
        os.makedirs(tmp_dir, exist_ok=True)

        episode = 1
        while True:
            filename = tmp_dir + f'/instance_{episode}.lp'
            graph = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=0, directed=True)  # seed=random.integers(2 ** 16),
            gen.generate_multicommodity_network_flow(graph, n_nodes, n_commodities, filename, rng)
            # blocks the process until a slot in the queue is available
            orders_queue.put([filename, random.integers(2 ** 31)])
            episode += 1

    elif problem == "setcover":
        density = 0.05
        max_coef = 100
        n_rows = 500 if transfer else 400
        n_cols = 1000 if transfer else 750
        tmp_dir = out_dir + f'/tmp_{n_rows}_{n_cols}_{density}'
        os.makedirs(tmp_dir, exist_ok=True)

        episode = 1
        while True:
            filename = tmp_dir + f'/instance_{episode}.lp'
            gen.generate_setcover(n_rows, n_cols, density, max_coef, filename, rng)
            # blocks the process until a slot in the queue is available
            orders_queue.put([filename, random.integers(2 ** 31)])
            episode += 1

    elif problem == "cauctions":
        n_items = 200 if transfer else 100
        n_bids = 1000 if transfer else 500
        tmp_dir = out_dir + f'/tmp_{n_items}_{n_bids}'
        os.makedirs(tmp_dir, exist_ok=True)

        episode = 1
        while True:
            filename = tmp_dir + f'/instance_{episode}.lp'
            gen.generate_cauctions(n_items, n_bids, filename, rng)
            # blocks the process until a slot in the queue is available
            orders_queue.put([filename, random.integers(2 ** 31)])
            episode += 1


def collect_solutions(problem, config, n_jobs, k_sols, random):
    """
    Runs branch-and-bound episodes on the given set of instances
    and collects the best k_solutions, which are saved to files.

    Parameters
    ----------
    problem : str
        Problem type
    config : dict
        Dictionary with configuration settings.
    n_jobs : int
        Number of jobs for parallel sampling.
    k_sols : int
        Number of solutions to save to file
    random: np.random.Generator
        Random number generator
    """
    in_queue = mp.JoinableQueue()
    out_queue = mp.SimpleQueue()

    workers = []
    for i in range(n_jobs):
        p = mp.Process(
            target=solve_instance,
            args=(in_queue, out_queue, k_sols),
            daemon=True)
        workers.append(p)
        p.start()

    instances = []
    difficulty = config['difficulty'][problem]
    for instance_type in ["train", "valid"]:
        instance_dir = f'data/{problem}/instances/{instance_type}_{difficulty}'
        instances += glob.glob(instance_dir + '/*.lp')

    for instance in instances:
        in_queue.put([instance, random.integers(2**31)])
    print(f"{len(instances)} instances on queue.")

    in_queue.join()
    obj_values = {}
    while not out_queue.empty():
        instance = out_queue.get()
        filename = instance['filename']
        opt_sol = instance['opt_sol']
        obj_values[filename] = opt_sol

    for p in workers:
        p.terminate()

    with open(f'data/{problem}/instances/obj_values.json', "w") as f:
        json.dump(obj_values, f)


def collector(problem, config, n_jobs, k_sols, random):
    in_queue = mp.JoinableQueue(2 * n_jobs)
    out_queue = mp.SimpleQueue()

    workers = []
    for i in range(n_jobs):
        p = mp.Process(
            target=solve_instance,
            args=(in_queue, out_queue, k_sols),
            daemon=True)
        workers.append(p)
        p.start()

    dispatcher = mp.Process(
        target=generate_instances,
        args=(in_queue, problem, random),
        daemon=True)
    dispatcher.start()

    tmp_dirs = []
    obj_values = {}

    nnodes = []
    for instance_type, num_instances in config['num_instances']:
        if instance_type == "transfer":
            # Stop the dispatcher
            dispatcher.terminate()
            # Empty the in_queue
            while not in_queue.empty():
                in_queue.get()
                in_queue.task_done()
            # Complete all running processes
            in_queue.join()
            # Discard the results
            while not out_queue.empty():
                out_queue.get()
            # start a new dispatcher
            dispatcher = mp.Process(
                target=generate_instances,
                args=(in_queue, problem, random, True),
                daemon=True)
            dispatcher.start()

        tmp_dir = None
        for i in trange(num_instances):
            instance = out_queue.get()
            old_filename = instance['filename']
            tmp_dir = os.path.dirname(old_filename)
            instance_dir = tmp_dir.replace('tmp', instance_type)
            os.makedirs(instance_dir, exist_ok=True)

            new_filename = instance_dir + f'/instance_{i + 1}.lp'
            os.rename(old_filename, new_filename)

            if instance_type in ["train", "valid"]:
                for j in range(instance['num_sols']):
                    os.rename(f'{old_filename[:-3]}-{j + 1}.sol',
                              f'{new_filename[:-3]}-{j + 1}.sol')
                nnodes.append(instance['nnodes'])

            obj_values[new_filename] = instance['opt_sol']

        if instance_type in ["test", "transfer"]:
            tmp_dirs.append(tmp_dir)

    # secure objective values as soon as possible to avoid losing data
    with open(f'data/{problem}/instances/obj_values.json', "w") as f:
        json.dump(obj_values, f)

    print({'nnodes_mean': np.mean(nnodes), 'nnodes_std': np.std(nnodes)})

    # stop all workers (hard)
    dispatcher.terminate()
    in_queue.join()
    for p in workers:
        p.terminate()
    print(tmp_dirs)
    for tmp_dir in tmp_dirs:
        shutil.rmtree(tmp_dir, ignore_errors=True)


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
        '-k', '--ksols',
        help='Number of solutions to save.',
        default=config['k'],
        type=int,
    )
    parser.add_argument(
        '-j', '--njobs',
        help='Number of parallel jobs.',
        default=1,
        type=int,
    )
    args = parser.parse_args()
    config['num_instances'] = [("train", 20),
                               ("valid", 10),
                               ("test", 1),
                               ("transfer", 1)]

    rng = np.random.default_rng(args.seed)
    if os.path.exists(f'data/{args.problem}/instances'):
        collect_solutions(args.problem, config, args.njobs, args.ksols, rng)
    else:
        collector(args.problem, config, args.njobs, args.ksols, rng)
