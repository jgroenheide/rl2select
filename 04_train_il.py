# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
# Train agent using the imitation learning method of Gasse et al.               #
# Output is saved to out_dir/<seed>_<timestamp>/best_params_il.pkl              #
# Usage: python 04_train_il.py <type> -s <seed> -g <cudaId>                     #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #

import os
import json
import glob
import time
import argparse
import model as ml
import numpy as np
import torch as th
import torch_geometric

from utilities import log
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from data import Dataset, GraphDataset


class Scheduler(th.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)
        self.step_result = -1

    def step(self, metrics, epoch=...):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        self.num_bad_epochs += 1
        self.step_result = -1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
            self.step_result = 0  # NEW_BEST
        elif self.num_bad_epochs == self.patience:
            for param_group in self.optimizer.param_groups:
                old_lr = float(param_group['lr'])
                new_lr = old_lr * self.factor
                if old_lr - new_lr > self.eps:
                    param_group['lr'] = new_lr
            self.step_result = 1  # NO_PATIENCE
        elif self.num_bad_epochs == 2 * self.patience:
            self.step_result = 2  # ABORT


def process(policy, data_loader, optimizer=None):
    avg_loss = 0
    avg_acc = 0
    num_samples = 0

    training = optimizer is not None
    with th.set_grad_enabled(training):
        for state, action in data_loader:
            # batch = batch.to(device)
            target = th.unsqueeze(action, -1).float()
            output = policy(*state)

            # Loss calculation for binary output
            loss = th.nn.BCELoss()(output, target)
            y_pred = th.round(output)

            # Loss calculation for 3+ output heads
            # loss = th.nn.CrossEntropyLoss()(output, target.long())
            # y_pred = th.argmax(output, dim=1)

            avg_loss += loss.item() * action.shape[0]
            avg_acc += th.sum(th.eq(y_pred, target)).item()
            num_samples += action.shape[0]

            if training:
                optimizer.zero_grad()
                loss.backward()  # Does backpropagation and calculates gradients
                optimizer.step()  # Updates the weights accordingly

    avg_loss /= max(num_samples, 1)
    avg_acc /= max(num_samples, 1)

    return avg_loss, avg_acc


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
        '-s', '--seed',
        help='Random generator seed.',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=-1,
    )
    parser.add_argument(
        '-k',
        help='Number of solutions to process',
        type=int,
        default=10,
    )
    # # add all config parameters as optional command-line arguments
    # for param, value in config.items():
    #     if param not in ['seed', 'gpu', 'k']:
    #         parser.add_argument(
    #             f'--{param}',
    #             type=type(value),
    #             default=argparse.SUPPRESS,
    #         )
    args = parser.parse_args()

    # override config with command-line arguments if provided
    args_config = {key: getattr(args, key) for key in config.keys() & vars(args).keys()}
    config.update(args_config)

    # --- HYPER PARAMETERS --- #
    model = "MLP"
    max_epochs = 1000
    batch_train = 32
    batch_valid = 128
    lr = 1e-3
    entropy_bonus = 0.0

    difficulty = config['difficulty'][args.problem]
    static = True

    # --- PYTORCH SETUP --- #
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = "cpu"
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = f"cuda:0"

    # --- LOG --- #

    # Create timestamp to save weights
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H.%M.%S')
    timestamp = f"{current_date}--{current_time}"
    running_dir = f'experiments/{args.problem}_{difficulty}/{args.seed}_{timestamp}'
    os.makedirs(running_dir, exist_ok=True)
    logfile = os.path.join(running_dir, 'il_train_log.txt')

    log(f"max_epochs: {max_epochs}", logfile)
    log(f"batch_size: {batch_train}", logfile)
    log(f"valid_batch_size : {batch_valid}", logfile)
    log(f"learning rate: {lr}", logfile)
    log(f"entropy bonus: {entropy_bonus}", logfile)
    log(f"problem: {args.problem}", logfile)
    log(f"gpu: {args.gpu}", logfile)
    log(f"seed {args.seed}", logfile)

    tensorboard_path = os.path.join(running_dir, f'tensorboard_log')
    writer = SummaryWriter(log_dir=tensorboard_path)

    # --- POLICY AND DATA --- #
    sample_dir = f'data/{args.problem}/samples/train_{difficulty}'
    train_files = [str(file) for file in glob.glob(sample_dir + '/sample_*.pkl')]
    sample_dir = f'data/{args.problem}/samples/valid_{difficulty}'
    valid_files = [str(file) for file in glob.glob(sample_dir + '/sample_*.pkl')]

    if model == "MLP":
        model = ml.MLPPolicy().to(device)

        train_data = Dataset(train_files)
        valid_data = Dataset(valid_files)
        train_loader = DataLoader(train_data, batch_train, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_valid, shuffle=False)
    elif model == "GNN":
        model = ml.GNNPolicy().to(device)

        train_data = GraphDataset(train_files)
        valid_data = GraphDataset(valid_files)

        follow_batch = ['constraint_features_s',
                        'constraint_features_t',
                        'variable_features_s',
                        'variable_features_t']

        train_loader = torch_geometric.loader.DataLoader(train_data, batch_train, shuffle=True, follow_batch=follow_batch)
        valid_loader = torch_geometric.loader.DataLoader(valid_data, batch_valid, shuffle=False, follow_batch=follow_batch)
    else:
        raise NotImplementedError

    optimizer = th.optim.Adam(model.parameters(), lr=lr)
    scheduler = Scheduler(optimizer, factor=0.2, patience=50)

    total_elapsed_time = 0
    epoch_loss = []
    epoch_acc = []
    for epoch in range(max_epochs + 1):
        start_time = time.time()

        # TRAIN
        train_loss, train_acc = process(model, train_loader, optimizer)
        log(f'Epoch {epoch} | train loss: {train_loss:.3f}, accuracy: {train_acc:.3f}', logfile)
        writer.add_scalar(f'{args.seed}/Loss/train', train_loss, epoch)
        writer.add_scalar(f'{args.seed}/Accuracy/train', train_acc, epoch)

        # TEST
        valid_loss, valid_acc = process(model, valid_loader)
        log(f'Epoch {epoch} | valid loss: {valid_loss:.3f}, accuracy: {valid_acc:.3f}', logfile)
        writer.add_scalar(f'{args.seed}/Loss/valid', valid_loss, epoch)
        writer.add_scalar(f'{args.seed}/Accuracy/valid', valid_acc, epoch)

        epoch_loss.append([train_loss, valid_loss])
        epoch_acc.append([train_acc, valid_acc])

        elapsed_time = time.time() - start_time
        total_elapsed_time += elapsed_time
        log(f"Epoch {epoch} | elapsed time: {elapsed_time:.3f} s | total: {total_elapsed_time:.3f} s", logfile)

        scheduler.step(valid_loss)
        if scheduler.step_result == 0:  # NEW_BEST
            log(f"Epoch {epoch} | found best model so far, valid_loss: {valid_loss:.3f}, acc: {valid_acc:.3f}", logfile)
            th.save(model.state_dict(), f'{running_dir}/best_params_il.pkl')
        elif scheduler.step_result == 1:  # NO_PATIENCE
            log(f'Epoch {epoch} | {scheduler.patience} epochs without improvement, lowering learning rate', logfile)
        elif scheduler.step_result == 2:  # ABORT
            log(f'Epoch {epoch} | no improvements for {2 * scheduler.patience} epochs, early stopping', logfile)
            break

    writer.close()

    model.load_state_dict(th.load(f'{running_dir}/best_params_il.pkl'))
    valid_loss, valid_acc = process(model, valid_loader)
    log(f"BEST VALID LOSS: {valid_loss:0.3f} | BEST VALID ACCURACY: {valid_acc:0.3f}", logfile)
