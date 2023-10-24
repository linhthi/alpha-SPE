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
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info

class CitationDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading SBM datasets
        """
        start = time.time()
        self.name = name
        self.mode = 'full'
        print("[I] Loading dataset %s..." % (name))
        self.save_path = 'data/processed/citation.{}/'.format(name)

        if not self.has_cache():
            self.create_and_save()
        else:
            self.load()
              
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
        self.graph.ndata['EigVecs'] = torch.Tensor(FullEigVecs)
        self.graph.ndata['EigVals'] = torch.Tensor(FullEigVals)
        print("[I] Finished adding structural features.")
        print("[I] Structural features time: {:.4f}s".format(time.time() - start))
    
    def create_and_save(self):
        dataset = None
        if self.name == 'CORA':
            dataset = CoraGraphDataset()
        elif self.name == 'CITESEER':
            dataset = CiteseerGraphDataset()
        elif self.name == 'PUBMED':
            dataset = PubmedGraphDataset()
        self.dataset = dataset
        graph = dataset[0]
        self.num_classes = dataset.num_classes
        self.graph = graph
        self.train = graph.ndata['train_mask']
        self.val = graph.ndata['val_mask']
        self.test = graph.ndata['test_mask']
        self.labels = graph.ndata['label']

        self.add_spe()
        self.graphs = [self.graph]
        self.labels = [self.labels]

        # save graphs and labels
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        save_graphs(graph_path, self.dataset, {'labels': self.labels})
        # save other information in python dict
        info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
        save_info(info_path, {'num_classes': self.num_classes})

    def load(self):
        # load processed data from directory `self.save_path`
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels']
        self.graph = self.graphs[0]
        self.train = self.graph.ndata['train_mask']
        self.val = self.graph.ndata['val_mask']
        self.test = self.graph.ndata['test_mask']
        self.labels = self.labels[0]
        info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
        self.num_classes = load_info(info_path)['num_classes']

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
        return os.path.exists(graph_path) and os.path.exists(info_path)
