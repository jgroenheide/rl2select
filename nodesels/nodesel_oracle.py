import utilities

from nodesels.nodesel_baseline import NodeselEstimate


# This class contains the sampler.
# Valid states reached by the BestEstimate selector are saved.
class NodeselOracle(NodeselEstimate):
    def __init__(self, sampler, sampling, solutions):
        super().__init__()
        self.sampler = sampler
        self.sampling = sampling
        self.solutions = solutions

        # save parent sol rank for speedup
        # root node is always an oracle node
        self.is_sol_node = {1: 0}

    def nodeselect(self):
        # Stop sampling after 5000 nodes
        if self.model.getNNodes() > 5000:
            self.model.interruptSolve()
        if self.sampling == "Nodes":
            print("*** ===================== ***")
            selnode = self.model.getBestNode()
            return {'selnode': selnode}

        depth = self.model.getDepth()
        if depth < 0:
            # choose the root node to start
            return super().nodeselect()

        node = self.model.getCurrentNode()
        node_number = node.getNumber()
        if node_number not in self.is_sol_node:
            # continue normal selection
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
            # If the parent node contained the optimal sol,
            # it is sufficient to only check the new bounds
            branchings = child.getParentBranchings()
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

        state = utilities.extract_MLP_state(self.model, *children)
        self.sampler.create_sample(*state, action)

        return super().nodeselect()

    def nodecomp(self, node1, node2):
        print(f"Node1: {node1.getNumber()} | Node2: {node2.getNumber()}")
        if self.sampling != "Nodes":
            return super().nodecomp(node1, node2)

        max_rank = len(self.solutions)
        sol_ranks = [max_rank, max_rank]
        for node_index, node in enumerate([node1, node2]):
            # If the parent node contained the optimal sol,
            # it is sufficient to only check the new bounds
            parent_number = node.getParent().getNumber()
            if parent_number not in self.is_sol_node: continue

            branchings = node.getParentBranchings()
            for sol_rank, sol in enumerate(self.solutions):
                for bvar, bbound, btype in zip(*branchings):
                    if btype == 0 and sol[bvar] < bbound: break  # EXCEEDS LOWER BOUND
                    if btype == 1 and sol[bvar] > bbound: break  # EXCEEDS UPPER BOUND
                else:                                            # SATISFIES ALL BOUNDS
                    node_number = node.getNumber()
                    self.is_sol_node[node_number] = 0
                    sol_ranks[node_index] = sol_rank
                    break  # break from solutions loop

        if sol_ranks[0] == sol_ranks[1]:
            return super().nodecomp(node1, node2)

        action = int(sol_ranks[1] < sol_ranks[0])
        both = sol_ranks[0] < max_rank and sol_ranks[1] < max_rank
        # parent_number = node.getParent().getNumber() if depth > 0 else 'ROOT' -> | Parent: {parent_number}
        print(f"Node1: {node1.getNumber()} | Node2: {node2.getNumber()} | Action: {['left', 'right'][action]} | Both: {both}")

        state = utilities.extract_MLP_state(self.model, node1, node2)
        self.sampler.create_sample(*state, action)

        return super().nodecomp(node1, node2)
