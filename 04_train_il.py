# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
# Train agent using the imitation learning method of Gasse et al.               #
# Output is saved to out_dir/<seed>_<timestamp>/best_params_il.pkl              #
# Usage: python 04_train_il.py <type> -s <seed> -g <cudaId>                     #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #

import os
import data
import json
import glob
import time
import argparse
import model as ml
import numpy as np
import torch as th
import wandb as wb
import torch_geometric

from utilities import log
from torch.utils.data import DataLoader


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
            target = action.float()
            output = policy(*state).squeeze()
            weight = th.where(action < 0.5, *norm_values)

            # Loss calculation for binary output
            loss = th.nn.BCELoss(weight)(output, target)
            y_pred = th.round(output)

            # Loss calculation for 3+ output heads
            # loss = th.nn.CrossEntropyLoss()(output, target.long())
            # y_pred = th.argmax(output, dim=1)

            if training:
                optimizer.zero_grad()
                loss.backward()  # Does backpropagation and calculates gradients
                optimizer.step()  # Updates the weights accordingly
                # wb.watch(policy)  # Save gradients to W&B

            avg_loss += loss.item() * action.shape[0]
            avg_acc += th.sum(th.eq(y_pred, target)).item()
            num_samples += action.shape[0]

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
        choices=config['problems']
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
    args = parser.parse_args()

    # --- HYPER PARAMETERS --- #
    model = "MLP"
    max_epochs = 10000
    batch_train = 32
    batch_valid = 128
    patience = 500
    lr = 5e-3

    # --- PYTORCH SETUP --- #
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = "cpu"
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = f"cuda:0"
    th.manual_seed(args.seed)

    # --- POLICY AND DATA --- #
    difficulty = config['difficulty'][args.problem]
    sample_dir = f'data/{args.problem}/samples/valid_{difficulty}'
    valid_files = [str(file) for file in glob.glob(sample_dir + '/sample_*.pkl')]
    sample_dir = f'data/{args.problem}/samples/train_{difficulty}'
    train_files = [str(file) for file in glob.glob(sample_dir + '/sample_*.pkl')]

    file_path = f"{sample_dir}/class_dist.json"
    if os.path.exists(file_path):
        # collect the pre-computed class distribution of the samples
        with open(file_path, "r") as f:
            class_dist = json.load(f)
    else: class_dist = [0.85, 0.15]
    # norm_values = [1 / x for x in class_dist]
    norm_values = [max(class_dist[1] / class_dist[0], 1),
                   max(class_dist[0] / class_dist[1], 1)]

    if model == "MLP":
        model = ml.MLPPolicy().to(device)

        train_data = data.Dataset(train_files)
        valid_data = data.Dataset(valid_files)
        train_loader = DataLoader(train_data, batch_train, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_valid, shuffle=False)
    elif model == "GNN":
        model = ml.GNNPolicy().to(device)

        train_data = data.GraphDataset(train_files)
        valid_data = data.GraphDataset(valid_files)

        follow_batch = ['constraint_features_s',
                        'constraint_features_t',
                        'variable_features_s',
                        'variable_features_t']

        train_loader = torch_geometric.loader.DataLoader(train_data, batch_train, shuffle=True, follow_batch=follow_batch)
        valid_loader = torch_geometric.loader.DataLoader(valid_data, batch_valid, shuffle=False, follow_batch=follow_batch)
    else:
        raise NotImplementedError

    optimizer = th.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = Scheduler(optimizer, factor=0.2, patience=patience)

    # --- LOG --- #

    # Create timestamp to save weights
    timestamp = time.strftime('%Y-%m-%d--%H.%M.%S')
    running_dir = f'experiments/{args.problem}_{difficulty}/{args.seed}_{timestamp}'
    os.makedirs(running_dir)
    logfile = os.path.join(running_dir, 'il_train_log.txt')
    wb.init(project="rl2select", config=config)

    log(f"training files: {len(train_files)}", logfile)
    log(f"validation files: {len(valid_files)}", logfile)
    log(f"batch size (train): {batch_train}", logfile)
    log(f"batch_size (valid): {batch_valid}", logfile)
    log(f"max epochs: {max_epochs}", logfile)
    log(f"learning rate: {lr}", logfile)
    log(f"problem: {args.problem}", logfile)
    log(f"gpu: {args.gpu}", logfile)
    log(f"seed {args.seed}", logfile)

    best_epoch = 0
    total_elapsed_time = 0
    for epoch in range(max_epochs + 1):
        log(f'** Epoch {epoch}', logfile)
        start_time = time.time()

        # TRAIN
        train_loss, train_acc = process(model, train_loader, optimizer)
        log(f'Epoch {epoch} | train loss: {train_loss:.3f} | accuracy: {train_acc:.3f}', logfile)
        wb.log({'train_loss': train_loss, 'train_acc': train_acc}, step=epoch)

        # TEST
        valid_loss, valid_acc = process(model, valid_loader)
        log(f'Epoch {epoch} | valid loss: {valid_loss:.3f} | accuracy: {valid_acc:.3f}', logfile)
        wb.log({'valid_loss': valid_loss, 'valid_acc': valid_acc}, step=epoch)

        elapsed_time = time.time() - start_time
        total_elapsed_time += elapsed_time
        log(f"Epoch {epoch} | elapsed time: {elapsed_time:.3f}s | total: {total_elapsed_time:.3f}s", logfile)

        scheduler.step(valid_loss)
        if scheduler.step_result == 0:  # NEW_BEST
            log(f"Epoch {epoch} | found best model so far, valid_loss: {valid_loss:.3f}, acc: {valid_acc:.3f}", logfile)
            th.save(model.state_dict(), f'{running_dir}/best_params_il.pkl')
            best_epoch = epoch
        elif scheduler.step_result == 1:  # NO_PATIENCE
            log(f'Epoch {epoch} | {scheduler.patience} epochs without improvement, lowering learning rate', logfile)
        elif scheduler.step_result == 2:  # ABORT
            log(f'Epoch {epoch} | no improvements for {2 * scheduler.patience} epochs, early stopping', logfile)
            break

    model.load_state_dict(th.load(f'{running_dir}/best_params_il.pkl'))
    valid_loss, valid_acc = process(model, valid_loader)
    log(f"PROCESS COMPLETED: BEST MODEL FOUND IN EPOCH {best_epoch}", logfile)
    log(f"BEST VALID LOSS: {valid_loss:0.3f} | BEST VALID ACCURACY: {valid_acc:0.3f}", logfile)
    th.save(model.state_dict(), f'actor/{args.problem}/il.pkl')
