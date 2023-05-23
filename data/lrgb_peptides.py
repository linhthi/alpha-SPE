import pandas
import tqdm
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
import pandas as pd
import torch.nn.functional as F

class LRGBDGLDataset(DGLDataset):
    def __init__(self, df, target_names, dataset='train', preload=False):
        self.df = df
        self.target_names = target_names
        self.preload = preload
        self.dataset = dataset
        super().__init__(name="lrgb")
        # self.process()

    def process(self):

        df = self.df
        target_names = self.target_names
        
        smiles_list = df['smiles']
        self.graphs = []
        self.labels = []

        # For each graph ID...
        #for graph_id in range(len(smiles)):
        if self.preload:
            tmp_file = "%s_precompute.pkl" % self.dataset
            with open(tmp_file, "rb") as f:
                output = pickle.load(f)
        else:
            output = {}

        for graph_id in tqdm.tqdm(range(len(smiles_list))):
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
            g.edata["feat"] = torch.FloatTensor(graph['edge_feat'])

            g = dgl.add_self_loop(g)

            if self.preload:
                se = np.array(output["%s_SE" % graph_id])
                eigvals = np.array(output["%s_EigVals" % graph_id])
                eigvecs = np.array(output["%s_EigVecs" % graph_id])

                g.ndata['SE'] = torch.FloatTensor(se)
                g.ndata['EigVals'] = torch.FloatTensor(eigvals)
                g.ndata['EigVecs'] = torch.FloatTensor(eigvecs)
            else:
                g.ndata['SE'] = pf.add_structural_feats(g, 3, 32)
                # Adding postional features: Eigen values and Eigen vectors
                FullEigVals, FullEigVecs = pf.laplace_decomp(g, num_nodes)

                sum_abs = np.sum(np.abs(FullEigVecs[:, :32]), axis=1)

                # FullEigVecs[sum_abs < 1e-6, :] += 1e-6
            
                #print(graph_id, FullEigVecs.shape, sum_abs.shape)
                #print(len(FullEigVecs[sum_abs<1e-5]))

                FullEigVecs = torch.FloatTensor(FullEigVecs)
                FullEigVecs = F.normalize(FullEigVecs, p=2, dim=1, eps=1e-12, out=None)
                after_nomal = FullEigVecs.numpy()

                sum_abs = np.sum(np.abs(after_nomal[:, :32]), axis=1)
                #print(graph_id, sum_abs.shape)
                #print('After normal: ', len(after_nomal[sum_abs<1e-5]))

                g.ndata['EigVals'] = torch.FloatTensor(FullEigVals)
                g.ndata['EigVecs'] = torch.FloatTensor(FullEigVecs[:, :32])

                output["%s_SE" % graph_id] = g.ndata['SE'].numpy().tolist()
                output["%s_EigVals" % graph_id] = g.ndata['EigVals'].numpy().tolist()
                output["%s_EigVecs" % graph_id] = g.ndata['EigVecs'].numpy().tolist()

            self.graphs.append(g)
            self.labels.append(torch.FloatTensor(y))

        # Convert the label list to tensor for saving
        # self.labels = torch.FloatTensor(self.labels)

        if not self.preload:
            with open("%s_precompute.pkl" % self.dataset, "wb") as f:
                pickle.dump(output, f)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

class LRGBDataset(torch.utils.data.Dataset):
    def __init__(self):
        """
            Loading
        """
    def __init__(self, csv_file="data/peptide_structure_normalized_dataset.csv",
                   mask_file="data/splits_random_stratified_peptide.pickle"):
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
        # df_train = df.iloc[mask["train"][:32]].copy().reset_index(drop=True)
        # df_val = df.iloc[mask["val"][:100]].copy().reset_index(drop=True)
        # df_test = df.iloc[mask["test"][:32]].copy().reset_index(drop=True)
        print("Got train data: ", df_train.shape)
        print("Got val data: ", df_val.shape)
        print("Got test data: ", df_test.shape)

        self.train = LRGBDGLDataset(df_train, target_names, 'train', True)
        self.val = LRGBDGLDataset(df_val, target_names, 'val', True)
        self.test = LRGBDGLDataset(df_test, target_names, 'test', True)

    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        #labels = torch.cat(labels).long()
        labels = torch.cat(labels)
        #print(labels.shape)
        #for idx, graph in enumerate(graphs):
        #    graphs[idx].ndata['feat'] = graph.ndata['feat'].float()
        #    graphs[idx].edata['feat'] = graph.edata['feat'].float()
            # Adding structural features
        #    graphs[idx].ndata['SE'] = pf.add_structural_feats(graph)
            # Adding postional features: Eigen values and Eigen vectors
        #    FullEigVals, FullEigVecs = pf.laplace_decomp(graph, graph.num_nodes())
        #    if FullEigVecs.shape[1] < 16:
        #        FullEigVecs = np.resize(FullEigVecs, (FullEigVecs.shape[0], 16))

        #    graphs[idx].ndata['EigVals'] = torch.Tensor(FullEigVals)
        #    graphs[idx].ndata['EigVecs'] = torch.Tensor(FullEigVecs[:, :16])
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels.reshape((-1, 11))

    # def _add_self_loops(self):
    #     # Add all self loops for training and validation and test
    #     for graph in self.train.graphs:

            
