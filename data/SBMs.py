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


class load_SBMsDataSetDGL(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 name,
                 split):

        self.split = split
        self.is_test = split.lower() in ['test', 'val']
        with open(os.path.join(data_dir, name + '_%s.pkl' % self.split), 'rb') as f:
            self.dataset = pickle.load(f)
        self.node_labels = []
        self.graph_lists = []
        self.n_samples = len(self.dataset)
        self._prepare()

    def _prepare(self):

        print("preparing %d graphs for the %s set..." % (self.n_samples, self.split.upper()))

        for data in self.dataset:

            node_features = data.node_feat
            edge_list = (data.W != 0).nonzero()  # converting adj matrix to edge_list

            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(node_features.size(0))
            g.ndata['feat'] = node_features.long()
            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())

            # adding edge features for Residual Gated ConvNet
            # edge_feat_dim = g.ndata['feat'].size(1) # dim same as node feature dim
            edge_feat_dim = 1  # dim same as node feature dim
            g.edata['feat'] = torch.ones(g.number_of_edges(), edge_feat_dim)

            self.graph_lists.append(g)
            self.node_labels.append(data.node_label)

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.node_labels[idx]


class SBMsDatasetDGL(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            TODO
        """
        start = time.time()
        print("[I] Loading data ...")
        self.name = name
        data_dir = 'data/SBMs'
        self.train = load_SBMsDataSetDGL(data_dir, name, split='train')
        self.test = load_SBMsDataSetDGL(data_dir, name, split='test')
        self.val = load_SBMsDataSetDGL(data_dir, name, split='val')
        print(self.train[0][0].ndata['feat'].shape)
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))


class SBMsDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading SBM datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        data_dir = 'data/SBMs/'
        with open(data_dir + name + '.pkl', "rb") as f:
            f = pickle.load(f)
            self.train = f[0]
            self.val = f[1]
            self.test = f[2]

        print('train, test, val sizes :', len(self.train), len(self.test), len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        labels = torch.cat(labels).long()
        for idx, graph in enumerate(graphs):
            graphs[idx].ndata['feat'] = graph.ndata['feat'].float()
            graphs[idx].edata['feat'] = graph.edata['feat'].float()
            # Adding structural features
            graphs[idx].ndata['SE'] = pf.add_structural_feats(graph)
            # Adding postional features: Eigen values and Eigen vectors
            FullEigVals, FullEigVecs = pf.laplace_decomp(graph, graph.num_nodes())
            graphs[idx].ndata['EigVals'] = torch.Tensor(FullEigVals)
            graphs[idx].ndata['EigVecs'] = torch.Tensor(FullEigVecs[:, :16])
            # print(graphs[idx].ndata['EigVecs'].shape)
            # print(graphs[idx].ndata['EigVecs'].shape)
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels

    def _laplace_decomp(self, max_freqs):
        self.train.graph_lists = [pf.laplace_decomp(g, max_freqs) for g in self.train.graph_lists]
        self.val.graph_lists = [pf.laplace_decomp(g, max_freqs) for g in self.val.graph_lists]
        self.test.graph_lists = [pf.laplace_decomp(g, max_freqs) for g in self.test.graph_lists]

    def _make_full_graph(self):
        self.train.graph_lists = [pf.make_full_graph(g) for g in self.train.graph_lists]
        self.val.graph_lists = [pf.make_full_graph(g) for g in self.val.graph_lists]
        self.test.graph_lists = [pf.make_full_graph(g) for g in self.test.graph_lists]

    def _add_edge_laplace_feats(self):
        self.train.graph_lists = [pf.add_edge_laplace_feats(g) for g in self.train.graph_lists]
        self.val.graph_lists = [pf.add_edge_laplace_feats(g) for g in self.val.graph_lists]
        self.test.graph_lists = [pf.add_edge_laplace_feats(g) for g in self.test.graph_lists]