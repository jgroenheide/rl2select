# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
# Train agent using the reinforcement learning method. User must provide a      #
# mode in {mdp, tmdp+DFS, tmdp+ObjLim}. The training parameters are read from   #
# config.json which is overridden by command line inputs, if provided.          #
# Usage: python 04_train_rl.py <type> -s <seed> -g <cudaId>                     #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #

import os
import json
import glob
import time

import shutil
import argparse
import numpy as np
import torch as th
import wandb as wb

from agent import AgentPool
from brain import Brain
from utilities import log
from scipy.stats.mstats import gmean

if __name__ == '__main__':
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
        'mode',
        help='Training mode.',
        choices=['mdp', 'tmdp+DFS', 'tmdp+ObjLim'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        default=config['seed'],
        type=int
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        default=config['gpu'],
        type=int
    )
    args = parser.parse_args()

    # configure gpu
    if config['gpu'] == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
        device = "cpu"
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f"{config['gpu']}"
        device = f"cuda:0"

    if args.gpu > -1:
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False
        print(f"Number of CUDA devices: {th.cuda.device_count()}")
        print(f"Active CUDA Device: {th.cuda.current_device()}")

    rng = np.random.default_rng(args.seed)
    th.manual_seed(args.seed)

    # data
    difficulty = config['difficulty'][args.problem]

    # recover training / validation instances and collect
    # the pre-computed optimal solutions for the instances
    instance_dir = f'data/{args.problem}/instances'
    train_files = [str(file).replace('\\', '/') for file in
                   glob.glob(instance_dir + f'/train_{difficulty}/*.lp')]
    valid_files = [str(file).replace('\\', '/') for file in
                   glob.glob(instance_dir + f'/valid_{difficulty}/*.lp')]

    if not os.path.exists(instance_dir + f'/obj_values.json'):
        print("obj_values.json not found. Creating from instance_solutions")
        obj_values = {}
        with open(instance_dir + f'/train_{difficulty}/instance_solutions.json') as f:
            obj_values.update(json.load(f))
        with open(instance_dir + f'/valid_{difficulty}/instance_solutions.json') as f:
            obj_values.update(json.load(f))
        with open(instance_dir + f'/obj_values.json', 'w') as f:
            json.dump(obj_values, f)
    with open(instance_dir + f'/obj_values.json') as f:
        opt_sols = json.load(f)

    valid_batch = [{'path': instance, 'seed': seed, 'sol': opt_sols[instance]}
                   for instance in valid_files[:config['num_valid_instances']]
                   for seed in range(config['num_valid_seeds'])]


    def train_batch_generator():
        sign = 1 if args.problem in ["cflp"] else -1
        while True:
            train_batches = [{'path': instance, 'seed': rng.integers(0, 2**31), 'sol': sign * opt_sols[instance]}
                             for instance in rng.choice(train_files, size=config['episodes_per_epoch'], replace=True)]
            yield train_batches


    batch_generator = train_batch_generator()

    # --- LOGGING --- #
    # logger = utilities.configure_logging()

    # Create timestamp to save weights
    timestamp = time.strftime('%Y-%m-%d--%H.%M.%S')
    experiment_dir = f'experiments/{args.problem}/04_train_rl'
    running_dir = experiment_dir + f'/{args.seed}_{timestamp}'
    os.makedirs(running_dir, exist_ok=True)
    logfile = running_dir + '/rl_train_log.txt'
    paramfile = running_dir + f'/best_params_rl-{args.mode}.pkl'
    # wb.init(project="rl2select", config=config)

    log(f"training instances: {len(train_files)}", logfile)
    log(f"validation instances: {len(valid_files)}", logfile)
    log(f"max epochs: {config['num_epochs']}", logfile)
    log(f"learning rate: {config['lr_train_rl']}", logfile)
    log(f"problem: {args.problem}", logfile)
    log(f"gpu: {args.gpu}", logfile)
    log(f"seed {args.seed}", logfile)

    brain = Brain(config, device)
    agent_pool = AgentPool(brain, config['num_agents'], config['time_limit'], args.mode)
    agent_pool.start()

    # Already start jobs  [CREATE]
    train_batch = next(batch_generator)
    sample_rate = config['sample_rate']
    t_next = agent_pool.start_job(train_batch, sample_rate, greedy=False, static=True)
    v_next = agent_pool.start_job(valid_batch, 0.0, greedy=True, static=True)
    t_samples, t_stats, t_queue, t_access = t_next
    _, v_stats, v_queue, v_access = v_next

    # training loop
    elapsed_time = 0
    start_time = time.time()
    best_tree_size = np.inf
    for epoch in range(config['num_epochs'] + 1):
        log(f"** Epoch {epoch}", logfile)
        epoch_data = {}

        # Allow preempted jobs to access policy  [START]
        # TRAINING #
        if epoch < config['num_epochs']:
            t_samples, t_stats, t_queue, t_access = t_next
            t_access.set()  # Give the training agents access to the policy!
            log(f"  {len(train_batch)} training jobs running (preempted)", logfile)
            # do not do anything with the samples or stats yet, we have to wait for the jobs to finish!
            # t_queue.join()  # force all training jobs to finish for debugging reasons
        else:
            log(f"  training skipped", logfile)
        # VALIDATION #
        if (epoch % config['valid_freq'] == 0) or (epoch == config['num_epochs']):
            _, v_stats, v_queue, v_access = v_next
            v_access.set()  # Give the validation agents access to the policy!
            log(f"  {len(valid_batch)} validation jobs running (preempted)", logfile)
            # do not do anything with the stats yet, we have to wait for the jobs to finish!
            # v_queue.join()  # force all validation jobs to finish for debugging reasons
        else:
            log(f"  validation skipped", logfile)

        # Start next epoch's jobs  [CREATE]
        # Get a new group of agents into position
        # TRAINING #
        if epoch + 1 < config['num_epochs']:
            train_batch = next(batch_generator)
            t_next = agent_pool.start_job(train_batch, sample_rate, greedy=False, static=True)
        # VALIDATION #
        if epoch + 1 <= config['num_epochs']:
            if ((epoch + 1) % config['valid_freq'] == 0) or ((epoch + 1) == config['num_epochs']):
                v_next = agent_pool.start_job(valid_batch, 0.0, greedy=True, static=True)

        # Validate the finished jobs [EVALUATE]
        # TRAINING #
        if epoch < config['num_epochs']:
            t_queue.join()  # wait for all training episodes to be processed
            log("  training jobs finished", logfile)
            log(f"  {len(t_samples)} training samples collected", logfile)
            t_losses = brain.update(t_samples)
            log("  model parameters were updated", logfile)

            t_nnodess = [s['info']['nnodes'] for s in t_stats]
            t_lpiterss = [s['info']['lpiters'] for s in t_stats]
            t_times = [s['info']['time'] for s in t_stats]

            wb.log({
                'train_nnodes_g': gmean(t_nnodess),
                'train_nnodes': np.mean(t_nnodess),
                'train_time': np.mean(t_times),
                'train_lpiters': np.mean(t_lpiterss),
                'train_nsamples': len(t_samples),
                'train_loss': t_losses['loss'],
                'train_reinforce_loss': t_losses['reinforce_loss'],
                'train_entropy': t_losses['entropy'],
            }, step=epoch)
        # VALIDATION #
        if (epoch % config['valid_freq'] == 0) or (epoch == config['num_epochs']):
            v_queue.join()  # wait for all validation episodes to be processed
            log("  validation jobs finished", logfile)

            v_nnodess = [s['info']['nnodes'] for s in v_stats]
            v_lpiterss = [s['info']['lpiters'] for s in v_stats]
            v_times = [s['info']['time'] for s in v_stats]

            assert np.all(np.array(v_nnodess) > 0)
            tree_size = gmean(v_nnodess)
            wb.log({
                'valid_nnodes_g': tree_size,
                'valid_nnodes': np.mean(v_nnodess),
                'valid_nnodes_max': np.amax(v_nnodess),
                'valid_nnodes_min': np.amin(v_nnodess),
                'valid_time': np.mean(v_times),
                'valid_lpiters': np.mean(v_lpiterss),
            }, step=epoch)

            if tree_size < best_tree_size:
                best_tree_size = tree_size
                log("Best parameters so far (1-shifted geometric mean), saving model.", logfile)
                brain.save(paramfile)

        # If time limit is hit, stop process
        elapsed_time = time.time() - start_time
        if int(elapsed_time / 86400) >= 6: break

    log(f"Done. Elapsed time: {elapsed_time}", logfile)
    os.makedirs(f'actor/{args.problem}', exist_ok=True)
    shutil.copy(paramfile, f'actor/{args.problem}/rl_{args.mode}.pkl')

    v_access.set()
    t_access.set()
    agent_pool.close()
