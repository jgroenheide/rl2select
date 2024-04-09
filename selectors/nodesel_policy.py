from pyscipopt import scip

import numpy as np
import torch as th

import utilities
from nodesel_baseline import NodeselRestartDFS, NodeselEstimate


class NodeselPolicy(scip.Nodesel):
    def __init__(self, policy, config_file, seed, device, on_both, prune_on_both, only_prune_steps):
        super().__init__()
        self.policy = policy
        self.config_file = config_file
        self.rng = np.random.RandomState(seed)
        self.device = device
        self.on_both = on_both
        self.prune_on_both = prune_on_both
        self.only_prune_steps = only_prune_steps

        self.config_preprocessing = None
        self.state_buffer = None
        self.n_iters = 0
        self.total_iters = 0
        self.on_no_children_iters = 0
        self.on_children_iters = 0

        self.in_no_children = False
        self.in_negative_depth = False

    def on_no_children(self):
        raise NotImplementedError()

    def on_negative_depth(self):
        raise NotImplementedError()

    def nodeselect(self):
        self.total_iters += 1
        # Check if we're in negative depth
        depth = self.model.getDepth()
        if depth < 0:
            self.in_negative_depth = True
            return self.on_negative_depth()

        # If we're not in negative depth...
        self.in_negative_depth = False
        _, children, _ = self.model.getOpenNodes()

        # Check if there are children
        if len(children) == 0:
            self.in_no_children = True
            self.on_no_children_iters += 1
            return self.on_no_children()

        # If there are children available...
        self.in_no_children = False
        self.on_children_iters += 1
        if len(children) == 1:
            node_number = self.model.getCurrentNode().getNumber()
            print(f'Num children == 1: {node_number} | depth: {depth}')
            # No choice to be made here
            # children[0].setPolicyPrune(False)
            return self.on_no_children()

        # If there is not 0 and not 1 child available, there has to be 2.
        assert (len(children) == 2)

        b, n1, n2, g = utilities.extract_MLP_state(self.model, *children)
        mlp_state0 = [b, n1, g]
        mlp_state1 = [b, n2, g]

        # self.config_preprocessing = apply_preprocessing.one_sample_dict(combined, self.config_preprocessing, self.config_file)

        input_tensor = th.tensor(list(mlp_state0)).to(self.device).float()
        input_tensor = th.unsqueeze(input_tensor, dim=0)
        with th.no_grad():
            self.policy.eval()
            output = self.policy(input_tensor)
            if output.shape[1] != 1:
                _, indices = output.topk(2)
                node_to_select = indices[0][0].item()
            else:
                output_item = th.nn.Sigmoid()(output).item()
                node_to_select = 1 if output_item > 0.5 else 0

        if node_to_select == 2:  # 'both'
            if self.on_both == 'prioChild':
                # Select priority child set by branching rule:
                selnode = self.model.getPrioChild()
                selnode_number = selnode.getNumber()
                children_numbers = [x.getNumber() for x in children]
                node_to_select = children_numbers.index(selnode_number)
            elif self.on_both == 'second':
                # Select second best:
                node_to_select = indices[0][1].item()
            elif self.on_both == 'random':
                # Select a random node
                node_to_select = self.rng.choice(2)
            else:
                raise Exception('Invalid on_both parameter')
            prune = self.prune_on_both
        else:
            prune = True

        if output.shape[1] != 1:
            children[1 - node_to_select].setPolicyScore(output[0, 1 - node_to_select])
            children[node_to_select].setPolicyScore(output[0, node_to_select])
        else:
            children[1 - node_to_select].setPolicyScore(1 - output_item)
            children[node_to_select].setPolicyScore(output_item)

        children[1 - node_to_select].setPolicyPrune(prune)
        children[node_to_select].setPolicyPrune(False)

        self.n_iters += 1
        return super().nodeselect()\
            if self.only_prune_steps\
            else {"selnode": children[node_to_select]}

    def nodecomp(self, node1, node2):
        score1 = node1.getPolicyScore()
        score2 = node2.getPolicyScore()
        if score1 == score2:
            return super().nodecomp(node1, node2)
        return 1 if score1 < score2 else -1


class NodeselPolicyRestartDFS(NodeselPolicy):
    def __init__(self, policy, config_file, seed, device, on_both, prune,
                 prune_on_both, only_prune_steps):
        super().__init__(policy, config_file, seed, device, on_both,
                         prune_on_both, only_prune_steps)
        self.p = NodeselRestartDFS(prune)

    def on_no_children(self):
        return self.p.nodeselect()

    def on_negative_depth(self):
        return self.p.nodeselect()

    def nodecomp(self, node1, node2):
        return self.p.nodecomp(node1, node2) \
            if self.in_no_children or self.in_negative_depth \
            else super().nodecomp(node1, node2)


class NodeselPolicyEstimate(NodeselPolicy):
    def __init__(self, policy, config_file, seed, device, on_both, prune,
                 prune_on_both, only_prune_steps):
        super().__init__(policy, config_file, seed, device, on_both,
                         prune_on_both, only_prune_steps)
        self.p = NodeselEstimate(prune)

    def on_no_children(self):
        return self.p.nodeselect()

    def on_negative_depth(self):
        return self.p.nodeselect()

    def nodecomp(self, node1, node2):
        return self.p.nodecomp(node1, node2) \
            if self.in_no_children or self.in_negative_depth \
            else super().nodecomp(node1, node2)


class NodeselPolicyScore(NodeselPolicy):
    def __init__(self, policy, config_file, seed, device, on_both, prune,
                 prune_on_both, only_prune_steps):
        super().__init__(policy, config_file, seed, device, on_both,
                         prune_on_both, only_prune_steps)
        self.p = NodeselEstimate(prune)

    def on_no_children(self):
        return {'selnode': self.model.getBestNode()}

    def on_negative_depth(self):
        return self.p.nodeselect()

    def nodecomp(self, node1, node2):
        return self.p.nodecomp(node1, node2) \
            if self.in_negative_depth \
            else super().nodecomp(node1, node2)
