import queue
import extract
import numpy as np
import torch as th
import pyscipopt as scip


class NodeselAgent(scip.Nodesel):
    def __init__(self, instance, opt_sol, seed, greedy, static, sample_rate, requests_queue):
        super().__init__()
        # self.model = model
        self.receiver_queue = queue.Queue()
        self.requests_queue = requests_queue
        self.instance = instance
        self.opt_sol = opt_sol
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.greedy = greedy
        self.static = static
        self.sample_rate = sample_rate
        self.tree_recorder = TreeRecorder() if sample_rate > 0 else None
        self.transitions = []
        self.penalty = 0

        self.iter_count = 0
        self.info = {
            'nnodes': 0,  # ecole.reward.NNodes().cumsum(),
            'lpiters': 0,  # ecole.reward.LpIterations().cumsum(),
            'time': 0,  # ecole.reward.SolvingTime().cumsum()
        }

    def nodeselect(self):
        # calculate minimal and maximal plunging depth
        min_plunge_depth = int(self.model.getMaxDepth() / 10)
        if self.model.getNStrongbranchLPIterations() > 2*self.model.getNNodeLPIterations():
            min_plunge_depth += 10

        max_plunge_depth = int(self.model.getMaxDepth() / 2)
        max_plunge_depth = max(max_plunge_depth, min_plunge_depth)
        max_plunge_quot = 0.25

        # check if we are within the maximal plunging depth
        plunge_depth = self.model.getPlungeDepth()
        selnode = self.model.getBestChild()
        if plunge_depth <= max_plunge_depth and selnode is not None:
            # get global lower and cutoff bound
            lower_bound = self.model.getLowerbound()
            cutoff_bound = self.model.getCutoffbound()

            # if we didn't find a solution yet,
            # the cutoff bound is usually very bad:
            # use 20% of the gap as cutoff bound
            if self.model.getNSolsFound() == 0:
                max_plunge_quot *= 0.2

            # check, if plunging is forced at the current depth
            # else calculate maximal plunging bound
            max_bound = self.model.infinity()
            if plunge_depth >= min_plunge_depth:
                max_bound = lower_bound + max_plunge_quot * (cutoff_bound - lower_bound)

            if selnode.getEstimate() < max_bound:
                return {'selnode': selnode}

        return {'selnode': self.model.getBestboundNode()}

    def nodecomp(self, node1, node2):
        if node1.getParent() != node2.getParent(): return 0

        GUB = self.model.getUpperbound()
        if self.static and self.model.isEQ(GUB, self.opt_sol):
            self.model.interruptSolve()

        state1, state2 = extract.extract_MLP_state(self.model, node1, node2)
        state = (th.tensor(state1, dtype=th.float32),
                 th.tensor(state2, dtype=th.float32))

        # send out policy requests
        self.requests_queue.put({'state': state,
                                 'greedy': self.greedy,
                                 'receiver': self.receiver_queue})
        action = self.receiver_queue.get()  # LEFT:0, RIGHT:1

        # self.penalty = self.model.getNNodes()  # For global tree size
        # self.penalty = self.model.getUpperbound()  # For primal bound improvement

        # For optimality-bound penalty
        focus_node = self.model.getCurrentNode()
        bound = focus_node.getLowerbound()
        sense = self.model.getObjectiveSense()
        if sense == "minimize":
            self.penalty += (bound > self.opt_sol)
        elif sense == "maximize":
            self.penalty += (bound < self.opt_sol)
        else:
            raise ValueError

        # collect transition samples if requested
        if self.sample_rate > 0:
            focus_node = self.model.getCurrentNode()
            self.tree_recorder.record_branching_decision(focus_node)
            if self.rng.random() < self.sample_rate:
                node_number = focus_node.getNumber()
                self.transitions.append({'state': state,
                                         'action': action,
                                         'penalty': self.penalty,
                                         'node_id': node_number,
                                         })

        self.info.update({
            'nnodes': self.model.getNNodes(),
            'lpiters': self.model.getNLPIterations(),
            'time': self.model.getSolvingTime()
        })

        self.iter_count += 1
        # avoid too large trees during training for stability
        if self.iter_count > 50000 and not self.greedy:
            self.model.interruptSolve()

        return 1 if action > 0.5 else -1


class TreeRecorder:
    """
    Records the branch-and-bound tree from a custom brancher.

    Every node in SCIP has a unique node ID. We identify nodes and their corresponding
    attributes through the same ID system.
    Depth groups keep track of groups of nodes at the same depth. This data structure
    is used to speed up the computation of the subtree size.
    """

    def __init__(self):
        self.tree = {}
        self.depth_groups = []

    def record_branching_decision(self, focus_node):
        parent_node = focus_node.getParent()
        node_number = focus_node.getNumber()
        parent_number = (0 if node_number == 1 else
                         parent_node.getNumber())
        self.tree[node_number] = parent_number
        # Add to corresponding depth group
        depth = focus_node.getDepth()
        if len(self.depth_groups) > depth:
            self.depth_groups[depth].append(node_number)
        else:
            self.depth_groups.append([node_number])

    def calculate_subtree_sizes(self):
        subtree_sizes = {node_number: 0 for node_number in self.tree.keys()}
        for group in self.depth_groups[::-1]:  # [::-1] reverses the list
            for node_number in group:
                subtree_sizes[node_number] += 1
                if node_number > 1:
                    parent_number = self.tree[node_number]
                    subtree_sizes[parent_number] += subtree_sizes[node_number]
        return subtree_sizes
