import numpy as np


def build_subgraph(adj, n_hops, node_idx):
    """Build subgraph by adding n-hop neighbors of node_idx.

    Args:
        adj: [N, N] adjacency matrix
        n_hops: number of hops to add
        node_idx: node index to add neighbors for

    Returns:
        subgraph: [N, N] subgraph adjacency matrix
    """
    subgraph = adj.copy()
    for _ in range(n_hops):
        subgraph = subgraph + subgraph @ adj
    subgraph = (subgraph > 0).astype(np.float32)
    return subgraph * adj  # only keep edges in original graph


def bandit_sampler_exp3(adj, n_hops, node_idx, n_samples, gamma=0.1):
    """Sample n_samples nodes from the subgraph of node_idx.

    Args:
        adj: [N, N] adjacency matrix
        n_hops: number of hops to add
        node_idx: node index to add neighbors for
        n_samples: number of samples to take
        gamma: parameter for exp3 algorithm

    Returns:
        samples: [n_samples] sampled node indices
    """
    subgraph = build_subgraph(adj, n_hops, node_idx)
    n_nodes = subgraph.shape[0]
    degrees = subgraph.sum(1)
    weights = np.ones(n_nodes) / n_nodes
    samples = []
    for _ in range(n_samples):
        # sample node
        node = np.random.choice(n_nodes, p=weights)
        samples.append(node)
        # update weights
        weights = weights * np.exp(-gamma * subgraph[node] / degrees[node])
        weights = weights / weights.sum()
    return samples


def bandit_sampler_dep_round(adj, n_hops, node_idx, n_samples):
    """Sample n_samples nodes from the subgraph of node_idx.

    Args:
        adj: [N, N] adjacency matrix
        n_hops: number of hops to add
        node_idx: node index to add neighbors for
        n_samples: number of samples to take

    Returns:
        samples: [n_samples] sampled node indices
    """
    subgraph = build_subgraph(adj, n_hops, node_idx)
    n_nodes = subgraph.shape[0]
    degrees = subgraph.sum(1)
    weights = np.ones(n_nodes) / n_nodes
    samples = []
    for _ in range(n_samples):
        # sample node
        node = np.random.choice(n_nodes, p=weights)
        samples.append(node)
        # update weights
        weights = weights * (1 - subgraph[node] / degrees[node])
        weights = weights / weights.sum()
    return samples