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
import precompute_features as pf

from dgl.data import BitcoinOTCDataset, FB15kDataset, FB15k237Dataset


class DGLDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading SBM datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        dataset = None
        if name == 'bitcoin':
            dataset = BitcoinOTCDataset()
        elif name == 'fb15k':
            dataset = FB15kDataset
        elif name == 'fb15k237':
            dataset = FB15kDataset
        graph = dataset[0]
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

        print('train, test, val sizes :', len(self.train), len(self.test), len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        labels = torch.cat(labels).long()
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels

    def _laplace_decomp(self, max_freqs):
        self.train = pf.laplace_decomp(self.train, max_freqs)
        self.val = pf.laplace_decomp(self.val, max_freqs)
        self.test = pf.laplace_decomp(self.test, max_freqs)

    def _make_full_graph(self):
        self.train = pf.make_full_graph(self.train)
        self.val = pf.make_full_graph(self.val)
        self.test = pf.make_full_graph(self.test)

    def _add_edge_laplace_feats(self):
        self.train = pf.add_edge_laplace_feats(self.train)
        self.val = pf.add_edge_laplace_feats(self.val)
        self.test = pf.add_edge_laplace_feats(self.test)
