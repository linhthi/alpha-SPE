import pandas
import numpy as np
import pickle
import os
os.environ['DGLBACKEND'] = 'pytorch'
import torch
import dgl
from dgl.data import DGLDataset
import ogb
from ogb import utils as ogb_utils
import data.precompute_features as pf

class LRGBDGLDataset(torch.utils.data.Dataset):
    def __init__(self, df, target_names):
        self.df = df
        self.target_names = target_names
        super().__init__(name="lrgb")
        self.process()

    def process(self):

        df = self.df
        target_names = self.target_names
        
        smiles_list = df['smiles']
        self.graphs = []
        self.labels = []

        # For each graph ID...
        #for graph_id in range(len(smiles)):
        for graph_id in range(len(smiles_list)):
            smiles = smiles_list[graph_id]
            y = df.iloc[graph_id][target_names]

            graph = ogb_utils.smiles2graph(smiles)
            
            # Find the edges as well as the number of nodes and its label.
            edges_of_id = graph['edge_index']
            src = edges_of_id[0]
            dst = edges_of_id[1]
            num_nodes = graph['num_nodes']
            # Create a graph and add it to the list of graphs and labels.
            g = dgl.graph((src, dst), num_nodes=num_nodes)
            g.ndata["feat"] = torch.FloatTensor(graph['node_feat'])
            g.edata["weight"] = torch.FloatTensor(graph['edge_feat'])
            self.graphs.append(g)
            self.labels.append(y)

        # Convert the label list to tensor for saving.
        self.labels = torch.FloatTensor(self.labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

class LRGBDataset(torch.utils.data.Dataset):
    def __init__(self):
        """
            Loading
        """
    def __init__(self, csv_file="/home/linhht/peptide_structure_normalized_dataset.csv",
                   mask_file="/home/linhht/splits_random_stratified_peptide.pickle"):
        print("Read raw csv data from: ", csv_file)
        df = pd.read_csv(csv_file)
        print("Got: ", df.shape)
        print("Preprocessed data: ")

        target_names = ['Inertia_mass_a', 'Inertia_mass_b', 'Inertia_mass_c',
                    'Inertia_valence_a', 'Inertia_valence_b',
                    'Inertia_valence_c', 'length_a', 'length_b', 'length_c',
                    'Spherocity', 'Plane_best_fit']
        
        df.loc[:, target_names] = df.loc[:, target_names].apply(
            lambda x: (x - x.mean()) / x.std(), axis=0)
    
        with open(mask_file, "rb") as f:
            mask = pickle.load(f)

        df_train = df.iloc[mask["train"]].copy().reset_index(drop=True)
        df_val = df.iloc[mask["val"]].copy().reset_index(drop=True)
        df_test = df.iloc[mask["test"]].copy().reset_index(drop=True)
        print("Got train data: ", df_train.shape)
        print("Got val data: ", df_val.shape)
        print("Got test data: ", df_test.shape)

        self.train = LRGBDGLDataset(df_train, target_names)
        self.val = LRGBDGLDataset(df_val, target_names)
        self.test = LRGBDGLDataset(df_test, target_names)

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
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels    