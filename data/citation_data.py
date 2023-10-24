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

              
        # print('train, test, val sizes :', len(self.train), len(self.test), len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

    def add_spe(self):
        # Add local structural to encoding (SE)
        start = time.time()
        print("[I] Adding structural features ...")
        self.graph.ndata['SE'] = pf.add_structural_feats(self.graph)

        # Add global structural to encoding (GSPE)
        FullEigVals, FullEigVecs = pf.laplace_decomp(self.graph, self.graph.num_nodes())
        self.graph.ndata['EigVecs'] = FullEigVecs
        self.graph.ndata['EigVals'] = FullEigVals
        print("[I] Finished adding structural features.")
        print("[I] Structural features time: {:.4f}s".format(time.time() - start))
  

