# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
# Generates training samples for imitation learning                             #
# Usage: python 03_generate_il_samples.py <problem> <type> -s <seed> -j <njobs> #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #

import csv
import json
import glob
import argparse
import numpy as np

from scipy.stats import gmean, gstd


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
        'running_dir',
        help='Directory containing csv results.',
    )
    args = parser.parse_args()

    experiment_dir = f'experiments/{args.problem}/05_evaluate'
    result_files = glob.glob(experiment_dir + f'/{args.running_dir}/*.csv')
    print(result_files)
    for result_file in result_files:
        with open(result_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)

            current_instance = None
            mean_nnodes = []
            mean_stimes = []
            nnodes = []
            stimes = []
            for row in reader:
                if current_instance is None:
                    current_instance = row['instance']
                if row['instance'] != current_instance:
                    mean_nnodes.append(np.mean(nnodes))
                    mean_stimes.append(np.mean(stimes))
                    current_instance = row['instance']
                    nnodes = []; stimes = []
                nnodes.append(int(row['nnodes']))
                stimes.append(float(row['stime']))
            mean_nnodes.append(np.mean(nnodes))
            mean_stimes.append(np.mean(stimes))
            print(f"result_file: {result_file}"
                  f"| nnodes: {gstd(mean_nnodes)}"
                  f"| stimes: {gstd(mean_stimes)}")
