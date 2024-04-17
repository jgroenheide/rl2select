# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
# Generates training samples for imitation learning                             #
# Usage: python 03_generate_il_samples.py <problem> <type> -s <seed> -j <njobs> #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #

import os
import glob
import json
import argparse
import utilities
import numpy as np
import pyscipopt as scip
import multiprocessing as mp

from nodesels.nodesel_oracle import NodeselOracle
from utilities import log


def make_samples(in_queue, out_queue, out_dir):
    """
    Worker loop: fetch an instance, run an episode and record samples.

    Parameters
    ----------
    in_queue: mp.Queue
        Instance files from which to collect samples.
    out_queue : queue.Queue
        Output queue in which to put solutions.
    out_dir : str
        Directory in which to write samples.
    """
    while True:
        # Fetch an instance...
        episode, instance, seed = in_queue.get()
        instance_id = f'[w {os.getpid()}] episode {episode}'
        print(f'{instance_id}: Processing instance \'{instance}\'...')

        # Retrieve available solution files
        solution_files = glob.glob(f'{instance[:-3]}-*.sol')
        print(f"{instance_id}: Retrieved {len(solution_files)} solutions")
        if len(solution_files) == 0:
            print("ABORT: No solutions")
            continue

        # Initialize SCIP model
        m = scip.Model()
        m.hideOutput()
        m.readProblem(instance)

        # 1: CPU user seconds, 2: wall clock time
        m.setIntParam('timing/clocktype', 1)
        m.setRealParam('limits/time', 300)
        utilities.init_scip_params(m, seed)

        solutions = []
        for solution_file in solution_files:
            solution = m.readSolFile(solution_file)
            solutions.append(solution)

        oracle = NodeselOracle(solutions, episode, out_queue, out_dir)

        m.includeNodesel(nodesel=oracle,
                         name='nodesel_oracle',
                         desc='BestEstimate node selector that saves samples based on a diving oracle',
                         stdpriority=999999,
                         memsavepriority=999999)

        out_queue.put({
            'type': 'start',
            'episode': episode,
            'instance': instance,
            'seed': seed,
        })

        m.optimize()
        m.freeProb()

        count = max(oracle.sample_count, 1)
        action_count = [f'{action / count:.2f}' for action in oracle.action_count]
        print(f'{instance_id}: {action_count}: {oracle.both_count / count:.2f}')
        print(f'{instance_id}: Process completed, {oracle.sample_count} samples')

        out_queue.put({
            'type': 'done',
            'episode': episode,
            'instance': instance,
            'seed': seed,
        })


def send_orders(orders_queue, instances, random):
    """
    Continuously send sampling orders to workers (relies on limited queue capacity).

    Parameters
    ----------
    orders_queue : mp.Queue
        Limited-size queue to which to send orders.
    instances : list
        Instance file names from which to sample episodes.
    random: np.random.Generator
        Random number generator
    """

    episode = 0
    while True:
        instance = random.choice(instances)
        # blocks the process until a free slot in the queue is available
        orders_queue.put([episode, instance, random.integers(2**31)])
        episode += 1


def collect_samples(instances, out_dir, random, n_jobs, max_samples):
    """
    Runs branch-and-bound episodes on the given set of instances, and collects
    randomly (state, action) pairs from the 'vanilla-fullstrong' expert brancher.

    Parameters
    ----------
    instances : list
        Instance files from which to collect samples.
    out_dir : str
        Directory in which to write samples.
    random: np.random.Generator
        Random number generator
    n_jobs : int
        Number of jobs for parallel sampling.
    max_samples : int
        Number of samples to collect.
    """
    os.makedirs(out_dir, exist_ok=True)

    # start workers
    # orders_queue = mp.Queue(maxsize=2*n_jobs)
    answers_queue = mp.SimpleQueue()

    # temp solution for limited threads
    orders_queue = mp.Queue()
    for episode, instance in enumerate(instances):
        orders_queue.put([episode, instance, random.integers(2**31)])
    print(f'{len(instances)} instances on queue.')

    workers = []
    for i in range(n_jobs):
        p = mp.Process(
            target=make_samples,
            args=(orders_queue, answers_queue, out_dir),
            daemon=True)
        workers.append(p)
        p.start()

    # start dispatcher
    # dispatcher = mp.Process(
    #     target=send_orders,
    #     args=(orders_queue, instances, random),
    #     daemon=True)
    # dispatcher.start()
    # print(f"[m {os.getpid()}] dispatcher started...")

    # record answers and write samples
    buffer = {}
    episode_i = 0
    n_samples = 0
    in_buffer = 0
    action_count = [0, 0]
    while n_samples < max_samples:
        sample = answers_queue.get()

        # add received sample to buffer
        if sample['type'] == 'start':
            # Create a new episode object
            buffer[sample['episode']] = []
        else:
            # Add samples to correct episode
            buffer[sample['episode']].append(sample)
            if sample['type'] == 'sample':
                in_buffer += 1

        # early stop dispatcher (hard)
        if in_buffer + n_samples >= max_samples:  # and dispatcher.is_alive():
            # dispatcher.terminate()
            print(f"[m {os.getpid()}] dispatcher stopped...")

        # if current_episode object is not empty...
        while episode_i in buffer and buffer[episode_i]:
            samples_to_write = buffer[episode_i]
            buffer[episode_i] = []

            # write samples from current episode
            for sample in samples_to_write:
                # if final sample is processed...
                if sample['type'] == 'done':
                    # move to next episode
                    del buffer[episode_i]
                    episode_i += 1
                    break

                # else write sample
                in_buffer -= 1
                n_samples += 1
                action_count[sample['action']] += 1
                # sample['filename'] = f'{out_dir}/tmp/sample_{episode}_{sample_count}.pkl'
                os.rename(sample['filename'], f'{out_dir}/sample_{n_samples}.pkl')
                print(f"[m {os.getpid()}] episode {sample['episode']}: "
                      f"{n_samples} / {max_samples} samples written ({in_buffer} in buffer).")

                # stop the episode as soon as
                # enough samples are collected
                if n_samples == max_samples:
                    buffer = {}
                    break

    # stop all workers (hard)
    for p in workers:
        p.terminate()

    class_dist = [f'{100 * x / max_samples:.1f}' for x in action_count]
    print(f"Sampling completed: (Left, Right): {class_dist}")
    with open(out_dir + "/class_dist.json", "w") as f:
        json.dump([x / max_samples for x in action_count], f)


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
        'instance_type',
        help='Type of instances to sample',
        choices=['train', 'valid'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=utilities.valid_seed,
        default=config['seed'],
    )
    parser.add_argument(
        '-j', '--njobs',
        help='Number of parallel jobs.',
        type=int,
        default=1,
    )
    parser.add_argument(
        '-r', '--ratio',
        help='Samples per instance ratio',
        type=int,
        default=10,
    )

    args = parser.parse_args()

    difficulty = config['difficulty'][args.problem]
    instance_dir = f'data/{args.problem}/instances/{args.instance_type}_{difficulty}'
    sample_dir = f'data/{args.problem}/samples/{args.instance_type}_{difficulty}'
    os.makedirs(sample_dir)  # create output directory, throws an error if it already exists
    # logfile = os.path.join(sample_dir, 'sample_log.txt')

    instances = glob.glob(instance_dir + '/*.lp')
    num_samples = args.ratio * len(instances)
    log(f"{len(instances)} {args.instance_type} instances for {num_samples} samples")

    rng = np.random.default_rng(args.seed)
    collect_samples(instances, sample_dir, rng, args.njobs, num_samples)
