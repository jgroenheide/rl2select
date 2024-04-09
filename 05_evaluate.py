# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
# Evaluate all GCNN models (il, mdp, tmdp+DFS, tmdp+ObjLim) and SCIP's default  #
# rule, on 2 benchmarks (test and transfer). Each instance-model pair is solved #
# with 5 different seeds. Output is written into a csv file.                    #
# Usage:                                                                        #
# python 05_evaluate.py <type> -g <cudaId>                                         #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #

import os
import csv
import json
import time
import argparse
import numpy as np
import pyscipopt as scip
import utilities
from selectors import nodesel_policy

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
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    args = parser.parse_args()

    time_limit = 3600

    seeds = [0, 1, 2, 3, 4]
    branching_policies = []

    # SCIP internal brancher baselines
    internal_branchers = ['relpscost']
    for brancher in internal_branchers:
        for seed in seeds:
            branching_policies.append({
                'type': 'internal',
                'name': brancher,
                'seed': seed,
            })
    # GCNN models
    gcnn_models = ['il', 'mdp', 'tmdp+DFS', 'tmdp+ObjLim']
    for model in gcnn_models:
        for seed in seeds:
            branching_policies.append({
                'type': 'gcnn',
                'name': model,
                'seed': seed,
            })

    print(f"problem: {args.problem}")
    print(f"gpu: {args.gpu}")
    print(f"time limit: {time_limit} s")

    ### PYTORCH SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = 'cpu'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = f"cuda:0"

    import torch as th
    import model as ml

    # load and assign pytorch models to policies (share models and update parameters)
    loaded_models = {}
    for policy in branching_policies:
        if policy['type'] == 'gcnn':
            if policy['name'] not in loaded_models:
                ### MODEL LOADING ###
                model = ml.GNNPolicy().to(device)
                model.load_state_dict(th.load(f"actor/{args.problem}/0/{policy['name']}.pkl"))
                policy['model'] = model

    print("running SCIP...")

    fieldnames = [
        'policy',
        'seed',
        'type',
        'instance',
        'nnodes',
        'nlps',
        'gap',
        'status',
        'stime',
        'walltime',
        'proctime',
    ]
    os.makedirs('results', exist_ok=True)

    instances = []
    difficulty = config['difficulty'][args.problem]
    transfer_difficulty = {
        "indset": "1000_4",
        "gisp": "80_0.5",
        "cflp": "60_35_5",
        "fcmcnf": "30_45_100",
        "setcover": "500_1000_0.05",
        "mknapsack": "100_12",
        "cauctions": "200_1000"
    }[args.problem]
    instance_dir = f"data/{args.problem}/instances"
    instances += [{'type': 'test', 'path': instance_dir + f"/test_{difficulty}/instance_{i + 1}.lp"} for i in range(40)]
    instances += [{'type': 'transfer', 'path': instance_dir + f"/transfer_{transfer_difficulty}/instance_{i + 1}.lp"} for i in range(40)]

    result_file = f"{args.problem}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    with open(f"results/{result_file}", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for instance in instances:
            print(f"{instance['type']}: {instance['path']}...")

            for policy in branching_policies:
                th.manual_seed(policy['seed'])

                m = scip.Model()
                m.setIntParam('display/verblevel', 0)
                m.readProblem(f"{instance['path']}")
                utilities.init_scip_params(m, seed=policy['seed'])
                m.setIntParam('timing/clocktype', 1)  # 1: CPU user seconds, 2: wall clock time
                m.setRealParam('limits/time', time_limit)

                brancher = nodesel_policy.NodeselPolicy(policy)
                m.includeBranchrule(
                    branchrule=brancher,
                    name=f"{policy['type']}:{policy['name']}",
                    desc=f"Custom PySCIPOpt branching policy.",
                    priority=666666, maxdepth=-1, maxbounddist=1)

                walltime = time.perf_counter()
                proctime = time.process_time()

                m.optimize()

                walltime = time.perf_counter() - walltime
                proctime = time.process_time() - proctime

                stime = m.getSolvingTime()
                nnodes = m.getNNodes()
                nlps = m.getNLPs()
                gap = m.getGap()
                status = m.getStatus()

                writer.writerow({
                    'policy': f"{policy['type']}:{policy['name']}",
                    'seed': policy['seed'],
                    'type': instance['type'],
                    'instance': instance['path'],
                    'nnodes': nnodes,
                    'nlps': nlps,
                    'gap': gap,
                    'status': status,
                    'stime': stime,
                    'walltime': walltime,
                    'proctime': proctime,
                })

                csvfile.flush()
                m.freeProb()

                print(f"  {policy['type']}:{policy['name']} {policy['seed']} --- "
                      f"{nnodes} nodes - {nlps} lps - {stime:.2f} ({walltime:.2f} wall {proctime:.2f} proc) s. - {status}")
