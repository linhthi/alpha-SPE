import time
import os
import pickle
import numpy as np

import dgl
import torch
import torch.nn.functional as F

from scipy import sparse as sp
import numpy as np
import networkx as nx

import hashlib
import data.precompute_features as pf
from dgl import LaplacianPE, RandomWalkPE

from dgl.data import CoraGraphDataset, CiteseerGraphDataset, RedditDataset, PubmedGraphDataset


class CitationDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading SBM datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        dataset = None
        if name == 'CORA':
            dataset = CoraGraphDataset()
        elif name == 'CITESEER':
            dataset = CiteseerGraphDataset()
        elif name == 'PUBMED':
            dataset = PubmedGraphDataset()
        graph = dataset[0]
        self.num_classes = dataset.num_classes
        self.graph = graph
        self.train = graph.ndata['train_mask']
        self.val = graph.ndata['val_mask']
        self.test = graph.ndata['test_mask']
        self.labels = graph.ndata['label']

        # Add normalized laplacian features
        degs = graph.in_degrees().float()
        norm_degs = torch.pow(degs, -0.5)
        norm_degs[torch.isinf(norm_degs)] = 0
        self.graph.ndata['norm_degs'] = norm_degs.unsqueeze(1)

        # Add LaplacianPE features (PE)
        # transform = LaplacianPE(k=3)
        # self.graph = transform(graph)
        
        # Add RandomWalkPE features
        # transform_rpe = RandomWalkPE(k=3)
        # self.graph = transform_rpe(graph)

        # Add local structural to encoding (SE)
        sampler = dgl.dataloading.ShaDowKHopSampler([10, 5, 5]) # using 3 hop
        dataloader = dgl.dataloading.DataLoader(graph, torch.arange(graph.num_nodes()), sampler, 
                                        batch_size=1, shuffle=False, drop_last=False)
        i=0
        SE = []
        for input_nodes, output_nodes, subgraph in dataloader:
            i+=1
            # extract eigen val and vector from subgraph and to contruct the structure of each node

            EigVals, EigVecs = pf.laplace_decomp(subgraph, 32)
            # print(EigVals.shape)
            SE.append(EigVals)

        # print(SPE[0])
        SE = np.asarray(SE)
        # print(SPE.shape)
        SE = torch.Tensor(SE)
        self.graph.ndata['SE'] = SE

        # Add global structural to encoding (GSPE)
        FullEigVals, FullEigVecs = pf.laplace_decomp(graph, graph.num_nodes())
        self.graph.EigVecs = FullEigVecs
        self.graph.EigVals = FullEigVals
        
        # print('train, test, val sizes :', len(self.train), len(self.test), len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        labels = torch.cat(labels).long()
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels
