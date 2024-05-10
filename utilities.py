import datetime
import observation
import numpy as np
import scipy as sp
import pyscipopt as scip


def valid_seed(seed):
    # Check whether seed is a valid random seed or not.
    # Valid seeds must be between 0 and 2**31 inclusive.
    seed = int(seed)
    if seed < 0 or seed > 2 ** 31:
        raise ValueError
    return seed


def log(log_message, logfile=None):
    out = f'[{datetime.datetime.now()}] {log_message}'
    print(out)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(out, file=f)


def init_scip_params(model, seed, static=False, presolving=True, heuristics=True, separating=True, conflict=True):
    seed = seed % 2147483648  # SCIP seed range

    if static:
        heuristics = False
        separating = False
        conflict = False

    # set up randomization
    model.setBoolParam('randomization/permutevars', True)
    model.setIntParam('randomization/permutationseed', seed)
    model.setIntParam('randomization/randomseedshift', seed)

    # disable separation and restarts during search
    model.setIntParam('separating/maxrounds', 0)
    model.setIntParam('presolving/maxrestarts', 0)

    # if asked, disable presolving
    if not presolving:
        model.setPresolve(scip.SCIP_PARAMSETTING.OFF)

    # if asked, disable primal heuristics
    if not heuristics:
        model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)

    # if asked, disable separating in the root (cuts)
    if not separating:
        model.setSeparating(scip.SCIP_PARAMSETTING.OFF)

    # if asked, disable conflict analysis (more cuts)
    if not conflict:
        model.setBoolParam('conflict/enable', False)

    # if first_solution_only:
    #     m.setIntParam('limits/solutions', 1)


def extract_GNN_state(model, buffer=None):
    """
    Compute a bipartite graph representation of the solver. In this representation,
    the variables and constraints of the MILP are the left- and right-hand side nodes,
    and an edge links two nodes iff the variable is involved in the constraint.
    Both the nodes and edges carry features.

    Parameters
    ----------
    model : scip.Model
        The current model.
    buffer : dict
        A buffer to avoid re-extracting redundant information from the solver.
    Returns
    -------
    variable_features : dictionary of type {'names': list, 'values': np.ndarray}
        The features associated with the variable nodes in the bipartite graph.
    edge_features : dictionary of type {'names': list, 'indices': np.ndarray, 'values': np.ndarray}
        The features associated with the edges in the bipartite graph.
        This is given as a sparse matrix in COO format.
    constraint_features : dictionary of type {'names': list, 'values': np.ndarray}
        The features associated with the constraint nodes in the bipartite graph.
    """
    if buffer is None or model.getNNodes() == 1:
        buffer = {}

    var_state = observation.variable_features(model, buffer.get('observation'))
    if 'state' in buffer:
        obj_norm = buffer['state']['obj_norm']
    else:
        obj_norm = np.linalg.norm(var_state['coefs'])
        obj_norm = 1 if obj_norm <= 0 else obj_norm

    # Column features / Variable features
    # ------------------------------------------
    variable_features = {
        'names': list(var_state.keys()),
        'values': list(var_state.values()),
    }

    # Row features / Constraint features
    # ------------------------------------------
    cons_state = observation.constraint_features(model, buffer.get('observation'))
    row_norms = cons_state['norms']
    row_norms[row_norms == 0] = 1

    if 'state' in buffer:
        row_feats = buffer['state']['row_feats']
        has_lhs = buffer['state']['has_lhs']
        has_rhs = buffer['state']['has_rhs']
    else:
        row_feats = {}
        has_lhs = np.nonzero(~np.isnan(cons_state['lhss']))[0]
        has_rhs = np.nonzero(~np.isnan(cons_state['rhss']))[0]
        row_feats['obj_cosine_similarity'] = np.concatenate((
            -cons_state['objcossims'][has_lhs],
            +cons_state['objcossims'][has_rhs]))
        row_feats['bias'] = np.concatenate((
            -(cons_state['lhss'] / row_norms)[has_lhs],
            +(cons_state['rhss'] / row_norms)[has_rhs]))

    row_feats['is_tight'] = np.concatenate((
        cons_state['is_at_lhs'][has_lhs],
        cons_state['is_at_rhs'][has_rhs]))

    row_feats['age'] = np.concatenate((
        cons_state['ages'][has_lhs],
        cons_state['ages'][has_rhs])) / (model.getNLPs() + 5)

    tmp = cons_state['dual_sols'] / (row_norms * obj_norm)
    row_feats['dual_sol_normalized'] = np.concatenate((
        -tmp[has_lhs],
        +tmp[has_rhs]))

    constraint_features = {
        'names': list(row_feats.keys()),
        'values': list(row_feats.values()),
    }

    # Edge features
    # ------------------------------------------
    if 'state' in buffer:
        edge_row_idxs = buffer['state']['edge_row_idxs']
        edge_col_idxs = buffer['state']['edge_col_idxs']
        edge_feats = buffer['state']['edge_feats']
    else:
        coef_matrix = sp.sparse.csr_matrix(
            (cons_state['coefs']['values'] / row_norms[cons_state['coefs']['row_ids']],
             (cons_state['coefs']['row_ids'], cons_state['coefs']['col_ids'])),
            shape=(len(cons_state['nnon_zeros']), len(var_state['coefs'])))
        coef_matrix = sp.sparse.vstack((
            -coef_matrix[has_lhs, :],
             coef_matrix[has_rhs, :])).tocoo(copy=False)

        edge_row_idxs, edge_col_idxs = coef_matrix.row, coef_matrix.col
        edge_feats = {'coef_normalized': coef_matrix.data, }

    edge_features = {
        'names': list(edge_feats.keys()),
        'values': list(edge_feats.values()),
        'indices': np.vstack([edge_row_idxs, edge_col_idxs]),
    }

    if 'state' not in buffer:
        buffer['state'] = {
            'obj_norm': obj_norm,
            'row_feats': row_feats,
            'has_lhs': has_lhs,
            'has_rhs': has_rhs,
            'edge_row_idxs': edge_row_idxs,
            'edge_col_idxs': edge_col_idxs,
            'edge_feats': edge_feats,
        }

    return variable_features, edge_features, constraint_features


def extract_MLP_state(model, node1, node2):
    max_depth = model.getDepth() + 1
    branch_state = observation.branching_features(model, node1)
    branch_state['n_inferences'] /= max_depth

    branching_features1 = list(branch_state.values())
    # branching_features = {
    #     'names': list(branch_state.keys()),
    #     'values': list(branch_state.values()),
    # }

    if node1.getParent() == node2.getParent():
        branching_features2 = branching_features1
    else:
        branch_state = observation.branching_features(model, node2)
        branch_state['n_inferences'] /= max_depth
        branching_features2 = list(branch_state.values())

    node_state1 = observation.node_features(model, node1)
    node_state2 = observation.node_features(model, node2)
    node_state1['relative_depth'] /= max_depth * 0.1  # / 0.1 <=> * 10
    node_state2['relative_depth'] /= max_depth * 0.1

    root_lb = abs(model.getRootNode().getLowerbound())
    if model.isZero(root_lb): root_lb = 0.0001
    node_state1['node_lb'] /= root_lb
    node_state1['estimate'] /= root_lb
    node_state2['node_lb'] /= root_lb
    node_state2['estimate'] /= root_lb

    global_state = observation.global_features(model)
    bound_norm = global_state['global_ub'] - global_state['global_lb']
    node_state1['relative_bound'] = (node_state1['node_lb'] - global_state['global_lb']) / bound_norm
    node_state2['relative_bound'] = (node_state2['node_lb'] - global_state['global_lb']) / bound_norm
    global_state['global_lb'] /= root_lb
    global_state['global_ub'] /= root_lb

    node_features1 = list(node_state1.values())
    # node_features1 = {
    #     'names': list(node_state1.keys()),
    #     'values': list(node_state1.values()),
    # }

    node_features2 = list(node_state2.values())
    # node_features2 = {
    #     'names': list(node_state2.keys()),
    #     'values': list(node_state2.values()),
    # }

    global_features = list(global_state.values())
    # global_features = {
    #     'names': list(global_state.keys()),
    #     'values': list(global_state.values()),
    # }

    state1 = np.concatenate((branching_features1, node_features1, global_features), dtype=np.float32)
    state2 = np.concatenate((branching_features2, node_features2, global_features), dtype=np.float32)
    # return branching_features1, branching_features2, node_features1, node_features2, global_features
    return state1, state2
