1. Generate instances | (data/{problem}/instances/{instance_type}_{difficulty}/instance_*.lp)
   (a) (Option) Use only one dataset for testing (split into train and validation later)
2. Generate solutions | (data/{problem}/instances/{instance_type}_{difficulty}/instance_*-*.sol)
   (a) Remove instances with less than k solutions.
   (b) (Option) Combine generation and solving step.
3. Generate samples   | (data/{problem}/samples/{instance_type}_{difficulty}/sample_*.pkl)
   (a) For each [instance, solutions] generate [state, action] pairs from the oracle.
   (b) The state should include both nodes of the comparison.
   (c) The action should be from [-1, 0, 1] for left, no preference, right.
   (b) Samples should encode both the GNN and the MLP state.
   (c) (Option) When 'both', save the node comparison as '0'.
   (d) (Option) Choose opposite of default nodesel.
4. Perform training   | (experiments/{problem}_{difficulty}/{seed}_{timestamp}/best_params_*-*.pkl)
   Train IL
   (a) MLP policy: [He: branching_features, node_features, global_features]
   (b) GNN policy: [Gasse: variable_features, edge_features, constraint_features]
   (c) Hybrid policy: [all of the above?]
   Train RL
   (a) Actor: MLP or GNN -> action [left, right]
5. Evaluate uses [test] and [transfer] instances
Since we only choose between the left and right child of a node, we implement a smart DFS node selector.
This means we can encode the environment as a TreeMDP, since we maintain temporal consistency.

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
   However, this eliminates the possibility of using the model as a critic in actor-critic RL
in Labassi et al. they calculate this for the two nodes of a nodecomp call.
=> [state, action] pairs must include both nodes of the comparison

For simplicity, generality, and extendability, it's probably best to make all models compare between two nodes,
and then call getBestChild() as the node selection policy to compare only the children.


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