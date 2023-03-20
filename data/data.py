"""
    File to load dataset based on user control from main file
"""
from data.SBMs import SBMsDataset
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, RedditDataset, PubmedGraphDataset
from data.citation_data import CitationDataset
from data.superpixels import SuperPixDataset, SuperPixDatasetDGL
from data.lrgb_peptides import LRGBDataset

def load_data(dataset, use_spe=False):
    """
        This function is called in the main_xx.py file
        returns:
        ; dataset object
    """
    # handling for MNIST or CIFAR Superpixels
    if dataset == 'MNIST' or dataset == 'CIFAR10':
        return SuperPixDataset(dataset)

    # handling for SBM datasets
    SBM_DATASETS = ['SBM_CLUSTER', 'SBM_PATTERN']
    if dataset in SBM_DATASETS:
        return SBMsDataset(dataset)

    if dataset == 'CORA' or dataset == 'CITESEER' or dataset == 'PUBMED':
        data = CitationDataset(dataset)
        if use_spe == True:
            data.add_spe()
        return data

    if dataset == 'REDDIT':
        return RedditDataset()

    if dataset == 'LRGB':
        return LRGBDataset()
    


