# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
# Generates training samples for imitation learning                             #
# Usage: python 03_generate_il_samples.py <problem> <type> -s <seed> -j <njobs> #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #

import os
import glob
import json
import queue
import argparse
import extract
import utilities
import numpy as np
import pyscipopt as scip

from nodesels.nodesel_oracle import NodeselOracle

import faulthandler
faulthandler.enable()


def make_samples(episode, instance, seed, tmp_dir, k_sols, sampling):
    out_queue = queue.Queue()

    instance_id = f'[w {os.getpid()}] episode {episode}'
    print(f"{instance_id}: Processing instance '{instance}'...")

    # Retrieve available solution files
    solution_files = sorted(glob.glob(f'{instance[:-3]}-*.sol'), key=len)[:k_sols]
    print(f"{instance_id}: Retrieved {len(solution_files)} solutions")
    if len(solution_files) == 0:
        print("ABORT: No solutions")
        return out_queue

    # Initialize SCIP model
    m = scip.Model()
    m.hideOutput()
    os.remove('samples_log.txt')
    m.setLogfile('samples_log.txt')
    m.readProblem(instance)

    # 1: CPU user seconds, 2: wall clock time
    m.setIntParam('timing/clocktype', 1)
    m.setRealParam('limits/time', 300)
    utilities.init_scip_params(m, seed)

    solutions = []
    for solution_file in solution_files:
        solution = m.readSolFile(solution_file)
        solutions.append(solution)

    if sampling == 'Weighted':
        sampler = extract.BaseSampler(episode, tmp_dir, out_queue)
    elif sampling == 'Random':
        sampler = extract.RandomSampler(episode, tmp_dir, out_queue)
    elif sampling == 'Double':
        sampler = extract.DoubleSampler(episode, tmp_dir, out_queue)
    else:
        raise ValueError

    oracle = NodeselOracle(sampler, solutions)

    m.includeNodesel(nodesel=oracle,
                     name='nodesel_oracle',
                     desc='BestEstimate node selector that saves samples based on a diving oracle',
                     stdpriority=999999,
                     memsavepriority=999999)

    m.optimize()
    m.freeProb()

    count = max(sampler.sample_count, 1)
    print(f"{instance_id}: {[f'{action / count:.2f}' for action in sampler.action_count]}")
    print(f"{instance_id}: Process completed, {sampler.sample_count} samples")

    out_queue.put({
        'type': "done",
        'episode': episode,
        'action_count': sampler.action_count,
        'sample_count': sampler.sample_count,
    })
    return out_queue


def collect_samples(instances, sample_dir, k_sols, max_samples, sampling, random):
    """
    Runs branch-and-bound episodes on the given set of instances,
    and collects (state, action) pairs from the diving oracle.

    Parameters
    ----------
    instances : list
        Instance files from which to collect samples.
    sample_dir : str
        Directory in which to write samples.
    max_samples : int
        Number of samples to collect.
    random: np.random.Generator
        Random number generator
    """
    tmp_dir = sample_dir + '/tmp'
    os.makedirs(tmp_dir, exist_ok=True)

    # record answers and write samples
    episode_i = 0
    n_samples = 0

    sample_count = 0
    action_count = [0, 0]
    while n_samples < max_samples:
        instance = random.choice(instances)
        seed = random.integers(0, 2**31)
        samples_to_write = make_samples(episode_i, instance, seed, tmp_dir, k_sols, sampling)

        # write samples from current episode
        for sample in list(samples_to_write.queue):
            # if final sample is processed...
            if sample['type'] == 'done':
                action_count[0] += sample['action_count'][0]
                action_count[1] += sample['action_count'][1]
                sample_count += sample['sample_count']
                print(f"[m {os.getpid()}] episode {sample['episode']}: "
                      f"{sample_count} / {max_samples} samples written.")
                episode_i += 1
                break

            # else write sample
            n_samples += 1
            os.rename(sample['filename'], f'{sample_dir}/sample_{n_samples}.pkl')

            # stop the episode as soon as
            # enough samples are collected
            if n_samples == max_samples:
                break

    class_dist = [f'{x / sample_count:.2f}' for x in action_count]
    print(f"Sampling completed: (Left, Right): {class_dist}")
    with open(sample_dir + '/class_dist.json', "w") as f:
        json.dump([x / sample_count for x in action_count], f)


if __name__ == '__main__':
    # read default config file
    with open('config.json') as f:
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
        'sampling_type',
        help='Type of instances to sample',
        choices=['Weighted', 'Random', 'Double'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=utilities.valid_seed,
        default=config['seed'],
    )
    parser.add_argument(
        '-k', '--ksols',
        help='Number of solutions to save.',
        default=config['k'],
        type=int,
    )
    parser.add_argument(
        '-r', '--ratio',
        help='Samples per instance ratio',
        type=int,
        default=10,
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    difficulty = config['difficulty'][args.problem]
    instance_dir = f'data/{args.problem}/instances/{args.instance_type}_{difficulty}'
    sample_dir = f'data/{args.problem}/samples/k={args.ksols}_{args.sampling_type}'

    instances = glob.glob(instance_dir + f'/*.lp')
    num_samples = args.ratio * len(instances)
    out_dir = sample_dir + f'/{args.instance_type}_{difficulty}'
    os.makedirs(out_dir, exist_ok=True)
    print(f"{len(instances)} {args.instance_type} instances for {num_samples} {args.sampling_type} samples")
    collect_samples(instances, out_dir, args.ksols, num_samples, args.sampling_type, rng)
