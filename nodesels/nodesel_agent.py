import queue
import utilities
import numpy as np
import torch as th
from nodesels.nodesel_baseline import NodeselDFS


class NodeselAgent(NodeselDFS):
    def __init__(self, instance, seed, opt_sol, greedy, sample_rate, requests_queue):
        super().__init__()
        # self.model = model
        self.opt_sol = opt_sol
        self.receiver_queue = queue.Queue()
        self.requests_queue = requests_queue
        self.instance = instance
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.greedy = greedy
        self.sample_rate = sample_rate
        self.tree_recorder = TreeRecorder() if sample_rate > 0 else None
        self.transitions = []
        self.reward = 0

        self.iter_count = 0
        self.info = {
            'nnodes': 0,  # ecole.reward.NNodes().cumsum(),
            'lpiters': 0,  # ecole.reward.LpIterations().cumsum(),
            'time': 0,  # ecole.reward.SolvingTime().cumsum()
        }

    def nodeselect(self):
        if self.model.getNChildren() != 2:
            # n = self.model.getNChildren()
            # print(f'return DFS node: {n}')
            return super().nodeselect()
        selnode = self.model.getBestChild()
        return {'selnode': selnode}

    def nodecomp(self, node1, node2):
        b, n1, n2, g = utilities.extract_MLP_state(self.model, node1, node2)
        state = (th.tensor(np.concatenate([b, n1, g]), dtype=th.float32),
                 th.tensor(np.concatenate([b, n2, g]), dtype=th.float32))

        # send out policy queries
        # should actions be chosen greedily w.r.t. the policy?
        self.requests_queue.put({'state': state,
                                 'greedy': self.greedy,
                                 'receiver': self.receiver_queue})
        action = self.receiver_queue.get()  # LEFT:0, RIGHT:1, BOTH:2
        reward = self.model.getNNodes()  # For global tree size
        reward = self.model.getUpperbound()  # For primal bound improvement

        # For optimality-bound reward
        focus_node = self.model.getCurrentNode()
        bound = focus_node.getLowerbound()
        sense = self.model.getObjectiveSense()
        if sense == "minimize":
            reward = (bound <= self.opt_sol) - 1
        elif sense == "maximize":
            reward = (bound >= self.opt_sol) - 1
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
                                         'reward': reward,
                                         'node_id': node_number,
                                         })

        self.reward = reward
        self.info.update({
            'nnodes': reward,
            'nlpiters': self.model.getNLPIterations(),
            'time': self.model.getSolvingTime()
        })

        self.iter_count += 1
        # avoid too large trees during training for stability
        if (self.iter_count > 50000) and not self.greedy:
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
        self.tree[node_number] = {'parent': parent_number}
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
                parent_number = self.tree[node_number]['parent']
                subtree_sizes[node_number] += 2  # shouldn't this be 1?
                if parent_number > 0:
                    subtree_sizes[parent_number] += subtree_sizes[node_number]
        return subtree_sizes
