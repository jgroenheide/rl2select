import datetime
import subprocess
import numpy as np
import wandb as wb
import pyscipopt as scip


def sweep():
    sweep_config = {
        'method': "random",
        'metric': {
            'name': "loss",
            'goal': "minimize",
        },
        'parameters': {
            'num_epochs': {'value': 50},
            'lr_train_rl': {
                'distribution': 'uniform',
                'min': 1e-6,
                'max': 1e-2,
            },
        }
    }

    def train(): subprocess.run(["python3", "04_train_rl.py", "gisp", "mdp"])
    sweep_id = wb.sweep(sweep_config, project="rl2select")
    wb.agent(sweep_id, train, count=5)


if __name__ == "__main__": sweep()


def valid_seed(seed):
    # Check whether seed is a valid random seed or not.
    # Valid seeds must be between 0 and 2**31 inclusive.
    seed = int(seed)
    if seed < 0 or seed > 2 ** 31:
        raise ValueError
    return seed


def log(log_message, logfile=None):
    out = f"[{datetime.datetime.now()}] {log_message}"
    print(out)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(out, file=f)


def init_scip_params(model, seed, static=False,
                     presolving=True, heuristics=True, separating=True, conflict=True, propagating=True):
    seed = seed % 2147483648  # SCIP seed range

    if static:
        model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)
        model.setSeparating(scip.SCIP_PARAMSETTING.OFF)
        model.setBoolParam('conflict/enable', False)
        # Most infeasible branching is the best static rule.
        # Note: static branching does not improve behaviour.
        # model.setIntParam('branching/mostinf/priority', 20000)

    # set up randomization
    model.setBoolParam('randomization/permutevars', True)
    model.setIntParam('randomization/permutationseed', seed)
    model.setIntParam('randomization/randomseedshift', seed)

    # disable separation and restarts during search
    model.setIntParam('separating/maxrounds', 0)
    model.setIntParam('presolving/maxrestarts', 0)

    # if asked, disable presolving
    if not presolving:
        model.setPresolve(scip.SCIP_PARAMSETTING.OFF)

    # if asked, disable primal heuristics
    if not heuristics:
        model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)

    # if asked, disable separating in the root (cuts)
    if not separating:
        model.setSeparating(scip.SCIP_PARAMSETTING.OFF)

    # if asked, disable conflict analysis (more cuts)
    if not conflict:
        model.setBoolParam('conflict/enable', False)

    # if asked, disable constraint propagation
    if not propagating:
        model.setIntParam("propagating/maxroundsroot", 0)
        model.setIntParam("propagating/maxrounds", 0)

    # if first_solution_only:
    #     m.setIntParam('limits/solutions', 1)


def extract_MLP_statistics(data_loader, num_samples):
    stats_min = np.zeros((16,))
    stats_max = np.zeros((16,))
    stats_avg = np.zeros((16,))
    for state, _ in data_loader:
        state = np.concatenate(state, axis=0)  # (2048, 12)
        stats_min = np.minimum(stats_min, state.min(axis=0))
        stats_max = np.maximum(stats_max, state.max(axis=0))
        stats_avg = np.add(stats_avg, state.sum(axis=0))
    stats_avg /= 2 * num_samples
    np.set_printoptions(suppress=True)
    print(stats_min)
    print(stats_max)
    print(stats_avg)
