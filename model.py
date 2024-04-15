import torch
import torch.nn.functional as F
import torch_geometric


class MLPPolicy(torch.nn.Module):

    def __init__(self):
        super().__init__()
        in_features = 17
        self.model = torch.nn.Sequential(torch.nn.Linear(in_features, 32),
                                         torch.nn.LeakyReLU(),
                                         torch.nn.Linear(32, 1))

    def forward(self, n0, n1):
        s0, s1 = self.model(n0), self.model(n1)
        return torch.sigmoid(-s0 + s1)


class GNNPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()

        emb_size = 32  # uniform node feature embedding dim

        hidden_dim1 = 8
        hidden_dim2 = 4
        hidden_dim3 = 4

        # static data
        cons_nfeats = 4
        edge_nfeats = 1
        var_nfeats = 6

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
        )

        self.bounds_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(2),
            torch.nn.Linear(2, 2),
            torch.nn.ReLU(),
        )

        # double check
        self.conv1 = torch_geometric.nn.GraphConv((emb_size, emb_size), hidden_dim1)
        self.conv2 = torch_geometric.nn.GraphConv((hidden_dim1, hidden_dim1), hidden_dim2)
        self.conv3 = torch_geometric.nn.GraphConv((hidden_dim2, hidden_dim2), hidden_dim3)

        self.convs = [self.conv1, self.conv2, self.conv3]

    def forward(self, batch, inv=False, epsilon=0.01):
        # create constraint masks. Constraints associated with variables
        # for which at least one of their bounds have changed
        # graph1 edges

        try:
            graph0 = (batch.constraint_features_s,
                      batch.edge_index_s,
                      batch.edge_attr_s,
                      batch.variable_features_s,
                      batch.bounds_s,
                      batch.constraint_features_s_batch,
                      batch.variable_features_s_batch)

            graph1 = (batch.constraint_features_t,
                      batch.edge_index_t,
                      batch.edge_attr_t,
                      batch.variable_features_t,
                      batch.bounds_t,
                      batch.constraint_features_t_batch,
                      batch.variable_features_t_batch)

        except AttributeError:
            graph0 = (batch.constraint_features_s,
                      batch.edge_index_s,
                      batch.edge_attr_s,
                      batch.variable_features_s,
                      batch.bounds_s)

            graph1 = (batch.constraint_features_t,
                      batch.edge_index_t,
                      batch.edge_attr_t,
                      batch.variable_features_t,
                      batch.bounds_t)

        if inv:
            graph0, graph1 = graph1, graph0

        # concatenation of averages variable/constraint features after conv
        score0 = self.forward_graph(*graph0)
        score1 = self.forward_graph(*graph1)
        return torch.sigmoid(-score0 + score1)

    def forward_graph(self, constraint_features, edge_indices, edge_features,
                      variable_features, bbounds, constraint_batch=None, variable_batch=None):

        # Assume edge indices var to cons, constraint_mask of shape [Nconvs]
        variable_features = self.var_embedding(variable_features)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        bbounds = self.bounds_embedding(bbounds)

        edge_indices_reversed = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        for conv in self.convs:
            # Var to cons
            constraint_features_next = F.relu(conv((variable_features, constraint_features),
                                                   edge_indices,
                                                   edge_weight=edge_features,
                                                   size=(variable_features.size(0), constraint_features.size(0))))

            # cons to var
            variable_features = F.relu(conv((constraint_features, variable_features),
                                            edge_indices_reversed,
                                            edge_weight=edge_features,
                                            size=(constraint_features.size(0), variable_features.size(0))))

            constraint_features = constraint_features_next

        if constraint_batch is not None:

            constraint_avg = torch_geometric.nn.pool.avg_pool_x(constraint_batch,
                                                                constraint_features,
                                                                constraint_batch)[0]
            variable_avg = torch_geometric.nn.pool.avg_pool_x(variable_batch,
                                                              variable_features,
                                                              variable_features)[0]
        else:
            constraint_avg = torch.mean(constraint_features, dim=0, keepdim=True)
            variable_avg = torch.mean(variable_features, dim=0, keepdim=True)

        return (torch.linalg.norm(variable_avg, dim=1) +
                torch.linalg.norm(constraint_avg, dim=1) +
                torch.linalg.norm(bbounds, dim=1))
