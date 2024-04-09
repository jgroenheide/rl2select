from pyscipopt import scip

import numpy as np
import torch as th

import utilities


class NodeselPolicy(scip.Nodesel):
    def __init__(self, policy, config_file, seed, device, backup):
        super().__init__()
        self.policy = policy
        self.config_file = config_file
        self.rng = np.random.RandomState(seed)
        self.device = device
        self.backup = backup

        self.config_preprocessing = None
        self.state_buffer = None
        self.n_iters = 0
        self.total_iters = 0
        self.on_no_children_iters = 0
        self.on_children_iters = 0

        self.in_no_children = False
        self.in_negative_depth = False

    def on_no_children(self):
        return self.backup.nodeselect()

    def on_negative_depth(self):
        return self.backup.nodeselect()

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

        self.n_iters += 1
        return {"selnode": children[node_to_select]}

    def nodecomp(self, node1, node2):
        if self.in_no_children or self.in_negative_depth:
            return self.backup.nodecomp(node1, node2)
        score1 = node1.getPolicyScore()
        score2 = node2.getPolicyScore()
        if score1 == score2:
            return super().nodecomp(node1, node2)
        return 1 if score1 < score2 else -1
