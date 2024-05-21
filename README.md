1. Generate instances | (data/{problem}/instances/{instance_type}_{difficulty}/instance_*.lp)
2. Generate solutions | (data/{problem}/instances/{instance_type}_{difficulty}/instance_*-*.sol)
   (a) Remove infeasible instances and instances with less than 100 nodes explored.
3. Generate samples   | (data/{problem}/samples/{instance_type}_{difficulty}/sample_*.pkl)
   (a) For each [instance, solutions] generate [state, action] pairs from the oracle.
   (b) The state should include both nodes of the comparison.
   (c) The action should be from [0, 1] for left, right.
   (d) (Option) Choose opposite of default nodesel.
4. Train model RL/IL  | (experiments/{problem}/04_train_il/{seed}_{timestamp}/best_params_il_{mode}.pkl)
                      | (actor/{problem}/{model_id}.pkl)
   (a) MLP policy: [He: branching_features, node_features, global_features]
4. Train model (RL)   | (experiments/{problem}/04_train_rl/{seed}_{timestamp}/best_params_rl_{mode}.pkl)
                      | (actor/{problem}/{model_id}.pkl)
5. Evaluate models    | (experiments/{problem}/05_evaluate/{seed}_{timestamp}/results.csv)
   (a) uses [test] and [transfer] instances


Compare reinforcement learning approach with imitation learning approach from Yilmaz paper
   (a) Imitation learning uses samples (state, action) that are found in the path from the root node to the k best solutions
      - This means solving the instances, taking the k best solutions, and then checking for each node whether it is
        on the path to at least one of the solutions. If yes, save the state and action chosen in that node.
   (b) Reinforcement learning uses transitions (state, action, reward, state') that are found from applying the policy.

Use exponential weighting yes or no?

The models predict the value of a leaf node.
In He et al. they calculate this for every open node.
In Yilmaz et al. they calculate this for the child nodes.
=> When we only consider children, the branching and tree features will largely overlap.
   the state can therefore best be described from the parent's perspective.
in Labassi et al. they calculate this for the two nodes of a nodecomp call.
=> [state, action] pairs must include both nodes of the comparison

Subtree size is not a good indication of decision quality, because good decisions can have large subtrees while bad decisions
can have small subtrees. Instead: Global tree size, Primal bound improvement, Optimality-Bound difference.
- Global tree size: (NNodes at transition - total NNodes after solving)
- Primal bound improvement: (New GUB - Old GUB) (might be distributed over subtrees)
- Optimality-Bound: -1 if LB > Opt else 0

Without subtree sizes, TreeMDP representation is useless. Don't have to stick to DFS. More freedom.
Variable selection can create multiple paths to the same optimal solution. Node selection can not.
Actor-Critic is not very promising, but maybe work out the idea about normalising the reward with global tree size.
Node selection is useless if all open nodes have a lower bound that's lower than the optimal solution value.
Only the order of nodes with a lower bound *between* the incumbent and the optimal solution value are interesting.
But also should not be explored because the optimal solution value can be found without them.
This is why primal difficult problems are more interesting for node selection, because once the optimal solution value is found,
the decision-making process is trivialised to solving all remaining nodes, which cannot be improved by node selection.

Combining instance generation and solving allows more control over the instances that are saved.
   Infeasible instances need to either be removed, or ignored during training.
   Instances that are solved in the root node by presolve can't be sampled from.
Idea: Dispatcher generates instances and adds them to the queue with a random seed attached.
      Solvers take the generated instances from the queue and solve them as usual.
      Collector takes the solved instances from tmp, renames and writes their solutions.
--
      Train, valid, and test can all use the same dispatcher. Collector should apply the correct name.
      Transfer instances would require a new dispatcher to be started. Can be in the same run.
      Writing solutions is done before name is changed. Send the model and write solutions in the collector.
Issue: Instances must be moved to the correct folder based on instance_type and difficulty, but difficulty is not available.
        - Encode the difficulty in the tmp folder name, or
        - use the hard-coded config file difficulties

```python
import importlib

gen = importlib.import_module('01_generate_instances.py')


def dispatcher(orders_queue, problem, random, transfer=False):
    out_dir = f'data/{problem}/instances/tmp_'

    edge_prob = 0.6
    drop_rate = 0.5
    n_nodes = 80 if transfer else 60

    episode = 0
    while True:
        filename = os.path.join(out_dir, f'instance_{episode}.lp')
        graph = gen.Graph.erdos_renyi(n_nodes, edge_prob, random)
        gen.generate_general_indset(graph, filename, drop_rate, random)
        # blocks the process until a free slot in the queue is available
        orders_queue.put([episode, filename, random.integers(2 ** 31)])
        episode += 1


def collector(problem, config, n_jobs, k_sols, random):
    orders_queue = mp.Queue(maxsize=2 * n_jobs)
    answers_queue = mp.SimpleQueue()

    workers = []
    for i in range(n_jobs):
        p = mp.Process(
            target=solver,
            args=(orders_queue, answers_queue, out_dir),
            daemon=True)
        workers.append(p)
        p.start()

    dispatcher = mp.Process(
        target=dispatcher,
        args=(orders_queue, problem, random),
        daemon=True)
    dispatcher.start()

    stats = {}
    for instance_type, num_instances in config['num_instances']:
        if instance_type == 'transfer':
            dispatcher.terminate()
            orders_queue = mp.Queue(maxsize=2 * n_jobs)
            answers_queue = mp.SimpleQueue()
            dispatcher = mp.Process(
                target=dispatcher,
                args=(orders_queue, problem, random, True),
                daemon=True)
            dispatcher.start()
        n_instances = 0
        while n_instances < num_instances:
            instance = answers_queue.get()
            tmp_dir = os.path.dirname(instance['filename'])
            instance_dir = tmp_dir.replace('tmp', instance_type)
            filename = instance_dir + f'/instance_{n_instances}.lp'
            os.rename(instance['filename'], filename)

            # retrieve and save solutions to individual files
            m = instance['model']
            solutions = m.getSols()[:k_sols]
            for i, solution in enumerate(solutions):
                m.writeSol(solution, f'{filename[:-3]}-{i + 1}.sol')
            stats.update({filename: m.getObjVal(), })
            m.freeProb()

    # stop all workers (hard)
    dispatcher.terminate()
    for p in workers:
        p.terminate()
```

```python
import numpy as np
import torch as th
sample_dir = f'data/{args.problem}/samples/{args.dir}/valid_{difficulty}'
sample_files = [str(file) for file in glob.glob(sample_dir + '/*.pkl')]

valid_data = data.Dataset(sample_files)
valid_loader = DataLoader(valid_data, 256)
stats_min = np.zeros((16,))
stats_max = np.zeros((16,))
for state, _ in valid_loader:
    # state.shape = [256, 16]
    state = np.concatenate(state, dim=1)
    state_min = state.min(dim=1)[0]
    stats_min = np.minimum(stats_min, state_min)
    stats_min.minimum(state_min)

    state_max = state.max(dim=1)[0]
    stats_max = np.maximum(stats_max, state_max)
```

Experiments:
IL:
- K sols: [1, 10]
- Problem: [GISP10, SSCFLP, CAUCTIONS]
    Total: 6 Experiments
RL:
- Problem: [GISP, CFLP]

Fixing the sampling code:
 - Either: Find out why PySCIPopt is crashing, fix the issue, then run the code as normal, or
           Make the sampler resistant to crashing. When a sub-process crashes, start a new one.
 - 