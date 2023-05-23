from scipy import sparse as sp
import numpy as np
import networkx as nx
import dgl
import torch
import torch.nn.functional as F
import dgl


def laplace_decomp(g, max_freqs):
    """
    Compute the eigendecomposition of the normalized laplacian of a graph
    :param g: DGL graph
    :param max_freqs: Number of eigenvectors to compute
    :return: Eigenvalues and Eigenvectors
    """
    # Compute the normalized laplacian
    n = g.num_nodes()
    A = g.adj().to_dense()
    N = torch.diag(torch.Tensor(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5))
    L = torch.eye(g.num_nodes()) - N * A * N

    # Compute the eigendecomposition
    EigVals, EigVecs = np.linalg.eigh(L.numpy())

    # Normalize and pad EigenVectors
    EigVals = np.sort(np.abs(np.real(EigVals)))

    if n < max_freqs:
        EigVals = np.pad(EigVals, (0, max_freqs - n), 'constant', constant_values=(1))
        EigVecs = np.pad(EigVecs, (0, max_freqs - n), 'constant', constant_values=(1))
    EigVals, EigVecs = EigVals[: max_freqs], EigVecs[:, :max_freqs]

    if EigVecs.shape[1] < 32:
        EigVecs = np.resize(EigVecs, (EigVecs.shape[0], 32))
        #np.append(EigVecs, np.ones((EigVecs.shape[0],16 - EigVecs.shape[1])),axis=1)

    return EigVals, EigVecs

def original_laplace_decomp(g, max_freqs):
    """
    Compute the eigendecomposition of the laplacian of a graph
    :param g: DGL graph
    :param max_freqs: Number of eigenvectors to compute
    :return: Eigenvalues and Eigenvectors
    """
    # Compute the Laplacian matrix
    n = g.num_nodes()
    A = g.adj().to_dense()
    D = torch.diag(torch.Tensor(dgl.backend.asnumpy(g.in_degrees())))
    L = D - A

    # Compute the eigendecomposition
    EigVals, EigVecs = np.linalg.eigh(L.numpy())

    # Normalize and pad EigenVectors
    EigVals = np.sort(np.abs(np.real(EigVals)))

    if n < max_freqs:
        EigVals = np.pad(EigVals, (0, max_freqs - n), 'constant', constant_values=(1))
        EigVecs = np.pad(EigVecs, (0, max_freqs - n), 'constant', constant_values=(1))
    
    EigVals, EigVecs = EigVals[: max_freqs], EigVecs[:, :max_freqs]
    return EigVals, EigVecs

def make_full_graph(g):
    full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))

    # Here we copy over the node feature data and laplace encodings
    full_g.ndata['feat'] = g.ndata['feat']

    try:
        full_g.ndata['EigVecs'] = g.ndata['EigVecs']
        full_g.ndata['EigVals'] = g.ndata['EigVals']
    except:
        pass

    # Populate edge features w/ 0s
    full_g.edata['feat'] = torch.zeros(full_g.number_of_edges(), dtype=torch.long)
    full_g.edata['real'] = torch.zeros(full_g.number_of_edges(), dtype=torch.long)

    # Copy real edge data over
    full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['feat'] = torch.ones(
        g.edata['feat'].shape[0], dtype=torch.long)
    full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['real'] = torch.ones(
        g.edata['feat'].shape[0], dtype=torch.long)

    return full_g


def add_edge_laplace_feats(g):
    EigVals = g.ndata['EigVals'][0].flatten()

    source, dest = g.find_edges(g.edges(form='eid'))

    # Compute diffusion distances and Green function
    g.edata['diff'] = torch.abs(g.nodes[source].data['EigVecs'] - g.nodes[dest].data['EigVecs']).unsqueeze(2)
    g.edata['product'] = torch.mul(g.nodes[source].data['EigVecs'], g.nodes[dest].data['EigVecs']).unsqueeze(2)
    g.edata['EigVals'] = EigVals.repeat(g.number_of_edges(), 1).unsqueeze(2)

    # No longer need EigVecs and EigVals stored as node features
    del g.ndata['EigVecs']
    del g.ndata['EigVals']

    return g


def add_structural_feats(g, k=3, k0=16):
    """
    Add structural features using k_hop extractor
    :param g:
    :param k: number of hops
    :param k0: number of eigenvalues to keep
    :return: g with structural features
    """
    # Add local structural to encoding (SE)
    sampler = dgl.dataloading.ShaDowKHopSampler([10, 5, 5]) # using 3 hop
    dataloader = dgl.dataloading.DataLoader(g, torch.arange(g.num_nodes()), sampler, 
                                    batch_size=1, shuffle=False, drop_last=False)
    SE = []
    for input_nodes, output_nodes, subgraph in dataloader:
        # extract eigen val and vector from subgraph and to contruct the structure of each node
        EigVals, EigVecs = original_laplace_decomp(subgraph, 32)
        SE.append(EigVals)
        
    SE = np.asarray(SE)
    if np.isnan(SE.any()):
        print('SE')

    SE = torch.FloatTensor(SE)
    return SE



