import utilities

from nodesels.nodesel_baseline import NodeselEstimate


# This class contains the sampler.
# Valid states reached by the BestEstimate selector are saved.
class NodeselOracle(NodeselEstimate):
    def __init__(self, sampler, solutions):
        super().__init__()
        self.sampler = sampler
        self.solutions = solutions

        # root node is always an oracle node
        self.is_sol_node = {1: 0}

    def nodeselect(self):
        depth = self.model.getDepth()
        if depth < 0:
            # continue normal selection
            return super().nodeselect()

        node = self.model.getCurrentNode()
        node_number = node.getNumber()
        if node_number not in self.is_sol_node:
            return super().nodeselect()
        # My children will be processed; my work is done
        k = self.is_sol_node[node_number]
        del self.is_sol_node[node_number]

        if self.model.getNChildren() < 2:
            return super().nodeselect()
        _, children, _ = self.model.getOpenNodes()

        max_rank = len(self.solutions)
        sol_ranks = [max_rank, max_rank]
        for child_index, child in enumerate(children):
            branchings = child.getParentBranchings()
            # By partitioning, it is sufficient to only check the new
            # bounds of the parent node and if sol satisfies the new bounds
            for sol_rank, sol in enumerate(self.solutions[k:]):
                for bvar, bbound, btype in zip(*branchings):
                    if btype == 0 and sol[bvar] < bbound: break  # EXCEEDS LOWER BOUND
                    if btype == 1 and sol[bvar] > bbound: break  # EXCEEDS UPPER BOUND
                else:                                            # SATISFIES ALL BOUNDS
                    child_number = child.getNumber()
                    self.is_sol_node[child_number] = k + sol_rank
                    sol_ranks[child_index] = k + sol_rank
                    break  # break from solutions loop

        # Save 'both' if both children lead to a solution
        action = int(sol_ranks[1] < sol_ranks[0])
        both = sol_ranks[0] < max_rank and sol_ranks[1] < max_rank
        # parent_number = node.getParent().getNumber() if depth > 0 else 'ROOT' -> | Parent: {parent_number}
        print(f"Node: {node_number} | Depth: {depth} | Action: {['left', 'right'][action]} | Both: {both}")

        b, n1, n2, g = utilities.extract_MLP_state(self.model, *children)
        state = ([b, n1, g], [b, n2, g])
        self.sampler.create_sample(*state, action)

        return super().nodeselect()
