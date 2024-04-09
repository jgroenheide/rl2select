import os
import gzip
import pickle
import utilities

from nodesels.nodesel_baseline import NodeselEstimate


# This class contains the sampler.
# Valid states reached by the BestEstimate selector are saved.
class NodeselOracle(NodeselEstimate):
    def __init__(self, solutions, episode, out_queue, out_dir):
        super().__init__()
        self.solutions = solutions
        self.episode = episode
        self.out_queue = out_queue
        self.out_dir = out_dir

        # root node is always an oracle node
        self.is_oracle_node = {1: 0}
        self.state_buffer = None
        self.action_counts = [0, 0, 0]
        self.sample_counter = 0

    def nodeselect(self):
        depth = self.model.getDepth()
        if depth < 0:
            # continue normal selection
            return super().nodeselect()

        node = self.model.getCurrentNode()
        node_number = node.getNumber()
        if node_number not in self.is_oracle_node:
            return super().nodeselect()
        # My children will be processed; my work is done
        k = self.is_oracle_node[node_number]
        del self.is_oracle_node[node_number]

        if self.model.getNChildren() < 2:
            return super().nodeselect()
        _, children, _ = self.model.getOpenNodes()

        sol_ranks = [-1, -1]
        for child_index, child in enumerate(children):
            branchings = child.getParentBranchings()
            # By partitioning, it is sufficient to only check the new
            # bounds of the parent node and if sol satisfies the new bounds
            for sol_rank, sol in enumerate(self.solutions[k:]):
                for bvar, bbound, btype in zip(*branchings):
                    if btype == 0 and sol[bvar] < bbound: break  # EXCEEDS LOWER BOUND
                    if btype == 1 and sol[bvar] > bbound: break  # EXCEEDS UPPER BOUND
                else:
                    child_number = child.getNumber()
                    self.is_oracle_node[child_number] = k + sol_rank  # SATISFIES ALL BOUNDS
                    sol_ranks[child_index] = k + sol_rank
                    break  # break from solutions loop

        # Both children are stubs. Abort.
        assert sol_ranks[0] != sol_ranks[1]

        # Save 'both' if both children lead to a solution
        action = int(sol_ranks[0] < sol_ranks[1])
        both = sol_ranks[0] > -1 and sol_ranks[1] > -1

        b, n1, n2, g = utilities.extract_MLP_state(self.model, *children)
        mlp_state0 = [b, n1, g]
        mlp_state1 = [b, n2, g]

        action_index = 2 if both else action
        self.action_counts[action_index] += 1
        label = ['left', 'right', 'both'][action_index]
        parent_number = node.getParent().getNumber() if depth > 0 else 'ROOT'
        print(f'Node: {node_number} | Parent: {parent_number} | Depth: {depth} | Action: {label}')

        os.makedirs(f'{self.out_dir}/tmp', exist_ok=True)
        filename = f'{self.out_dir}/tmp/sample_{self.episode}_{self.sample_counter}.pkl'

        with gzip.open(filename, 'wb') as f:
            f.write(pickle.dumps({
                'state0': mlp_state0,
                'state1': mlp_state1,
                'action': action,
                'both': both
            }))

        self.out_queue.put({
            'type': 'sample',
            'episode': self.episode,
            'filename': filename,
        })

        self.sample_counter += 1

        return super().nodeselect()
