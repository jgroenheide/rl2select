# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
# Train agent using the reinforcement learning method. User must provide a      #
# mode in {mdp, tmdp+DFS, tmdp+ObjLim}. The training parameters are read from   #
# config.json which is overriden by command line inputs, if provided.   #
# Usage: python 04_train_rl.py <type> -s <seed> -g <cudaId>                     #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #

import os
import json
import glob
import numpy as np
import wandb as wb
import argparse

from tqdm import tqdm
from datetime import datetime
from scipy.stats.mstats import gmean
from utilities import log

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

    # override config with command-line arguments if provided
    args_config = {key: getattr(args, key) for key in config.keys() & vars(args).keys()}
    config.update(args_config)

    # configure gpu
    if config['gpu'] == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = "cpu"
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f"{config['gpu']}"
        device = f"cuda:0"

    # import torch after gpu configuration
    import torch as th
    from brain import Brain
    from agent import AgentPool

    if args.gpu > -1:
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False
        print(f"Number of CUDA devices: {th.cuda.device_count()}")
        print(f"Active CUDA Device: {th.cuda.current_device()}")

    rng = np.random.RandomState(args.seed)
    th.manual_seed(args.seed)

    # data
    difficulty = config['difficulty'][args.problem]
    maximization_probs = ['cauctions', 'indset', 'mkapsack']

    # recover training / validation instances
    instance_dir = f'data/{args.problem}/instances/valid_{difficulty}'
    valid_instances = [str(file) for file in glob.glob(instance_dir + '/instance_*.lp')]
    instance_dir = f'data/{args.problem}/instances/train_{difficulty}'
    train_instances = [str(file) for file in glob.glob(instance_dir + '/instance_*.lp')]

    # collect the pre-computed optimal solutions for the training instances
    with open(f"{instance_dir}/instance_solutions.json", "r") as f:
        train_sols = json.load(f)

    valid_batch = [{'path': instance, 'seed': seed}
                   for instance in valid_instances
                   for seed in range(config['num_valid_seeds'])]


    def train_batch_generator():
        eps = -0.1 if args.problem in maximization_probs else 0.1
        train_batches = [{'path': instance, 'seed': rng.randint(0, 2 ** 31), 'sol': train_sols[instance] + eps}
                         for instance in rng.choice(train_instances, size=config['num_episodes_per_epoch'], replace=True)]
        while True:
            yield train_batches


    batch_generator = train_batch_generator()

    # --- LOGGING --- #
    # logger = utilities.configure_logging()

    # Create timestamp to save weights
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H.%M.%S')
    timestamp = f"{current_date}--{current_time}"
    running_dir = f'experiments/{args.problem}_{difficulty}/{args.seed}_{timestamp}'
    os.makedirs(running_dir, exist_ok=True)
    logfile = os.path.join(running_dir, 'rl_train_log.txt')
    wb.init(project="rl2select", config=config)

    log(f"training instances: {len(train_instances)}", logfile)
    log(f"validation instances: {len(valid_instances)}", logfile)
    # log(f"max epochs: {max_epochs}", logfile)
    # log(f"batch size (train): {batch_train}", logfile)
    # log(f"batch_size (valid): {batch_valid}", logfile)
    # log(f"learning rate: {lr}", logfile)
    log(f"problem: {args.problem}", logfile)
    log(f"gpu: {args.gpu}", logfile)
    log(f"seed {args.seed}", logfile)

    brain = Brain(config, device)
    agent_pool = AgentPool(brain, config['num_agents'], config['time_limit'], args.mode)
    agent_pool.start()

    # Already start jobs
    train_batch = next(batch_generator)
    t_next = agent_pool.start_job(train_batch, sample_rate=config['sample_rate'], greedy=False, block_policy=True)
    v_next = agent_pool.start_job(valid_batch, sample_rate=0.0, greedy=True, block_policy=True)

    # training loop
    start_time = datetime.now()
    best_tree_size = np.inf
    for epoch in tqdm(range(config['num_epochs'] + 1)):
        log(f'** Epoch {epoch}', logfile)
        epoch_data = {}

        # Allow preempted jobs to access policy  [START]
        # VALIDATION #
        if (epoch % config['validate_every'] == 0) or (epoch == config['num_epochs']):
            _, v_stats, v_queue, v_access = v_next
            v_access.set()  # Give the validation agents access to the policy!
            log(f"  {len(valid_batch)} validation jobs running (preempted)", logfile)
            # do not do anything with the stats yet, we have to wait for the jobs to finish !
            # v_queue.join()  # force all validation jobs to finish for debugging reasons
        else:
            log(f"  validation skipped", logfile)
        # TRAINING #
        if epoch < config['num_epochs']:
            t_samples, t_stats, t_queue, t_access = t_next
            t_access.set()  # Give the training agents access to the policy!
            log(f"  {len(train_batch)} training jobs running (preempted)", logfile)
            # do not do anything with the samples or stats yet, we have to wait for the jobs to finish !
            # t_queue.join()  # force all training jobs to finish for debugging reasons
        else:
            log(f"  training skipped", logfile)

        # Start next epoch's jobs  [CREATE]
        # Get a new group of agents into position
        # VALIDATION #
        if ((epoch + 1) % config['validate_every'] == 0) or ((epoch + 1) == config['num_epochs']):
            v_next = agent_pool.start_job(valid_batch, sample_rate=0.0, greedy=True, block_policy=True)
        # TRAINING #
        if epoch + 1 < config['num_epochs']:
            train_batch = next(batch_generator)
            t_next = agent_pool.start_job(train_batch, sample_rate=config['sample_rate'], greedy=False, block_policy=True)

        # VALIDATION #  [EVALUATE]
        if (epoch % config['validate_every'] == 0) or (epoch == config['num_epochs']):
            v_queue.join()  # wait for all validation episodes to be processed
            log('  validation jobs finished', logfile)

            v_nnodess = [s['info']['nnodes'] for s in v_stats]
            v_lpiterss = [s['info']['lpiters'] for s in v_stats]
            v_times = [s['info']['time'] for s in v_stats]

            epoch_data.update({
                'valid_nnodes_g': gmean(np.asarray(v_nnodess) + 1) - 1,
                'valid_nnodes': np.mean(v_nnodess),
                'valid_nnodes_max': np.amax(v_nnodess),
                'valid_nnodes_min': np.amin(v_nnodess),
                'valid_time': np.mean(v_times),
                'valid_lpiters': np.mean(v_lpiterss),
            })
            if epoch == 0:
                v_nnodes_0 = epoch_data['valid_nnodes'] if epoch_data['valid_nnodes'] != 0 else 1
                v_nnodes_g_0 = epoch_data['valid_nnodes_g'] if epoch_data['valid_nnodes_g'] != 0 else 1
            epoch_data.update({
                'valid_nnodes_norm': epoch_data['valid_nnodes'] / v_nnodes_0,
                'valid_nnodes_g_norm': epoch_data['valid_nnodes_g'] / v_nnodes_g_0,
            })

            if epoch_data['valid_nnodes_g'] < best_tree_size:
                best_tree_size = epoch_data['valid_nnodes_g']
                log('Best parameters so far (1-shifted geometric mean), saving model.', logfile)
                brain.save(os.path.join(running_dir, f"best_params_rl-{args.mode}.pkl"))

        # TRAINING #
        if epoch < config['num_epochs']:
            t_queue.join()  # wait for all training episodes to be processed
            log('  training jobs finished', logfile)
            log(f"  {len(t_samples)} training samples collected", logfile)
            t_losses = brain.update(t_samples)
            log('  model parameters were updated', logfile)

            t_nnodess = [s['info']['nnodes'] for s in t_stats]
            t_lpiterss = [s['info']['lpiters'] for s in t_stats]
            t_times = [s['info']['time'] for s in t_stats]

            epoch_data.update({
                'train_nnodes_g': gmean(t_nnodess),
                'train_nnodes': np.mean(t_nnodess),
                'train_time': np.mean(t_times),
                'train_lpiters': np.mean(t_lpiterss),
                'train_nsamples': len(t_samples),
                'train_loss': t_losses.get('loss', None),
                'train_reinforce_loss': t_losses.get('reinforce_loss', None),
                'train_entropy': t_losses.get('entropy', None),
            })

        wb.log(epoch_data, step=epoch)

        # If time limit is hit, stop process
        elapsed_time = datetime.now() - start_time
        if elapsed_time.days >= 6: break

    log(f"Done. Elapsed time: {elapsed_time}", logfile)

    v_access.set()
    t_access.set()
    agent_pool.close()
