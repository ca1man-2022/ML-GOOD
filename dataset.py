from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import scipy
import scipy.io
from sklearn.preprocessing import label_binarize
import torch_geometric.transforms as T
import pdb
import random
import pandas as pd
from torch.nn.functional import pad

from data_utils import even_quantile_labels, to_sparse_tensor

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, Twitch, PPI, Yelp
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.utils import stochastic_blockmodel_graph, subgraph, homophily
from os import path

def load_dataset(args):
    '''
    dataset_ind: in-distribution training dataset
    dataset_ood_tr: ood-distribution training dataset as ood exposure
    dataset_ood_te: a list of ood testing datasets or one ood testing dataset
    '''
    # multi-graph datasets, use one as ind, the other as ood
    if args.dataset == 'twitch':
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_twitch_dataset(args.data_dir)
    elif args.dataset == 'ppi':
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_ppi_dataset(args.data_dir)
    
    # single graph, use original as ind, modified graphs as ood
    elif args.dataset == 'dblp':
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_dblp_dataset(args.data_dir, args.ood_type)
    elif args.dataset == 'pcg':
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_pcg_dataset(args.data_dir, args.ood_type)
    elif args.dataset == 'humloc':
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_humloc_dataset(args.data_dir, args.ood_type)
    elif args.dataset == 'eukloc':
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_eukloc_dataset(args.data_dir, args.ood_type)
    elif args.dataset == 'humeuk':
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_hum_euk_dataset(args.data_dir)
    elif args.dataset == 'ppipcg':
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_ppi_pcg_dataset(args.data_dir)
    elif args.dataset == 'yelp':
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_yelp_dataset(args.data_dir, args.ood_type, args.subset_percentage)

    # single graph, use partial nodes as ind, others as ood according to domain info
    elif args.dataset in 'arxiv':
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_arxiv_dataset(args.data_dir)

    elif args.dataset in 'proteins':
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_proteins_dataset(args.data_dir, args.subset_percentage)
        # print("ind:", dataset_ind)
        # print("ood_tr:", dataset_ood_tr)
        # print("ood_te:", dataset_ood_te)
        # print(dataset_ind.edge_index)
        # pdb.set_trace()

    # single graph, use original as ind, modified graphs as ood
    elif args.dataset in ('cora', 'citeseer', 'pubmed', 'amazon-photo', 'amazon-computer', 'coauthor-cs', 'coauthor-physics'):
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_graph_dataset(args.data_dir, args.dataset, args.ood_type)

    else:
        raise ValueError('Invalid dataname')
    return dataset_ind, dataset_ood_tr, dataset_ood_te

def load_twitch_dataset(data_dir):
    transform = T.NormalizeFeatures()
    subgraph_names = ['DE', 'EN', 'ES', 'FR', 'RU']
    train_idx, valid_idx = 0, 1
    dataset_ood_te = []
    for i in range(len(subgraph_names)):
        torch_dataset = Twitch(root=f'{data_dir}Twitch',
                              name=subgraph_names[i], transform=transform)
        dataset = torch_dataset[0]
        dataset.node_idx = torch.arange(dataset.num_nodes)
        if i == train_idx:
            dataset_ind = dataset
        elif i == valid_idx:
            dataset_ood_tr = dataset
        else:
            dataset_ood_te.append(dataset)

    return dataset_ind, dataset_ood_tr, dataset_ood_te

def load_ppi_dataset(data_dir):
    transform = T.NormalizeFeatures()
    torch_dataset = PPI(root=f'{data_dir}PPI', transform=transform)
    dataset_ind = []
    dataset_ood_tr = []
    dataset_ood_te = []
    train_idx, valid_idx = 4, 8
    for i in range(len(torch_dataset)):
        dataset = torch_dataset[i]
        dataset.node_idx = torch.arange(dataset.num_nodes)
        if i < train_idx:
            dataset_ind.append(dataset)
        elif i < valid_idx:
            dataset_ood_tr.append(dataset)
        else:
            dataset_ood_te.append(dataset)

    return dataset_ind, dataset_ood_tr, dataset_ood_te


def load_yelp_dataset(data_dir, ood_type, subset_percentage):
    transform = T.NormalizeFeatures()
    dataset = Yelp(root=f'{data_dir}yelp', transform=transform)
    data = dataset[0]
    # x = data.x
    n = data.num_nodes
    # edge_index = torch.as_tensor(data.edge_index)
    # label = torch.as_tensor(data.y)
    data.node_idx = torch.arange(n)
    
    # center_node_mask = torch.zeros_like(data.node_idx)
    subset_size = int(subset_percentage * n)
    subset_indices = torch.randperm(n)[:subset_size]
    # center_node_mask[subset_indices] = True   
    dataset_ind = Data(
        x = data.x[subset_indices],
        edge_index=data.edge_index,
        y=data.y[subset_indices],
        train_mask=data.train_mask[subset_indices],
        val_mask=data.val_mask[subset_indices],
        test_mask=data.test_mask[subset_indices],
        node_idx=torch.arange(subset_size)
    )
    # subset_indices = random.sample(range(n), subset_size)
    # subset_indices_tensor = torch.tensor(subset_indices)
    
    # pdb.set_trace()
    # sub_edge_index, _ = subgraph(center_node_mask, edge_index)
    dataset_ood_tr = dataset_ind
    dataset_ood_te = dataset_ind
    # Data(x=x, edge_index=sub_edge_index, y=label)
    # dataset_ind.node_idx = data.node_idx[center_node_mask]
    
    
    if ood_type == 'feature':
        dataset_ood_tr = create_feat_noise_dataset(dataset_ood_tr)
        dataset_ood_te = create_feat_noise_dataset(dataset_ood_te)

    return dataset_ind, dataset_ood_tr, dataset_ood_te


'''
PPI[0] as in-distribution dataset,
PCG as out-of-distribution dataset.
'''
def load_ppi_pcg_dataset(data_dir):
    transform = T.NormalizeFeatures()
    dataset_in = PPI(root=f'{data_dir}PPI', transform=transform)
    # single graph is enough
    d_in = dataset_in[0]
    d_in.node_idx = torch.arange(d_in.num_nodes)
    # # multi-graph, so much that OOD detection
    # d_in = []
    # for i in range(len(dataset_in)):
    #     dataset = dataset_in[i]
    #     dataset.node_idx = torch.arange(dataset.num_nodes)
    #     d_in.append(dataset)
        
    dataset_out = PCGDataset(root='../data/pcg_50')
    d_out = dataset_out[0]
    d_out.node_idx = torch.arange(d_out.num_nodes)

    dataset_ind = d_in
    dataset_ood_te = d_out
    dataset_ood_tr = d_in

    return dataset_ind, dataset_ood_tr, dataset_ood_te

class DBLPDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DBLPDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['dblp_labelhash.txt', 'dblp.edgelist', 'labels.txt', 'features.txt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download your raw data files here
        pass

    def process(self):
        # 读取 dblp.edgelist 文件
        with open(self.raw_paths[1], 'r') as file:
            edges = [list(map(int, line.strip().split())) for line in file]

        # 读取 labels.txt 文件
        with open(self.raw_paths[2], 'r') as file:
            labels = [list(map(float, line.strip().split(','))) for line in file]

        # 读取 features.txt 文件
        with open(self.raw_paths[3], 'r') as file:
            features = [list(map(float, line.strip().split(','))) for line in file]

        labels_dict = {i: label for i, label in enumerate(labels)}

        # Build the PyG Data object
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.tensor(features, dtype=torch.float)

        num_nodes = len(features)
        num_classes = len(labels_dict[0])
        y = torch.zeros((num_nodes, num_classes), dtype=torch.float)

        # Fill in the one-hot encoded labels for each node
        for i in range(num_nodes):
            y[i] = torch.tensor(labels_dict.get(i, [0.0] * num_classes), dtype=torch.float)

        # Create a PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, y=y)

        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
    def len(self):
        return 1  # Assuming one graph in the dataset

    def get(self, idx):
        return self.data

def load_dblp_dataset(data_dir, ood_type):
    dataset = DBLPDataset(root='../data/dblp')
    data = dataset[0]
    data.node_idx = torch.arange(data.num_nodes)

    dataset_ind = data
    dataset_ood_tr = data
    dataset_ood_te = data 

    if ood_type == 'feature':
        dataset_ood_tr = create_feat_noise_dataset(data)
        dataset_ood_te = create_feat_noise_dataset(data)
    

    return dataset_ind, dataset_ood_tr, dataset_ood_te

class PCGDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(PCGDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['edges_undir.csv', 'features.csv', 'labels.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download your raw data files here
        pass

    def process(self):

        edges_df = pd.read_csv(self.raw_paths[0], header=None, names=['source', 'target'])
        features_df = pd.read_csv(self.raw_paths[1], sep=',', header=None)
        labels_df = pd.read_csv(self.raw_paths[2], sep=',', header=None)

        # Build the PyG Data object
        edge_index = torch.tensor(edges_df.values, dtype=torch.long).t().contiguous()
        # Pad the features to increase the dimension from 32 to 50
        x_padded = pad(torch.tensor(features_df.values, dtype=torch.float), (0, 50 - 32))
        # x = torch.tensor(features_df.values, dtype=torch.float)
        y = torch.tensor(labels_df.values, dtype=torch.float)

        # Create a PyTorch Geometric Data object
        data = Data(x=x_padded, edge_index=edge_index, y=y)
        # data = Data(x=x, edge_index=edge_index, y=y)

        torch.save((self.collate([data])), self.processed_paths[0])


def load_pcg_dataset(data_dir, ood_type):
    dataset = PCGDataset(root='../data/pcg')
    data = dataset[0]
    data.node_idx = torch.arange(data.num_nodes)

    dataset_ind = data
    dataset_ood_tr = data
    dataset_ood_te = data 

    if ood_type == 'feature':
        dataset_ood_tr = create_feat_noise_dataset(data)
        dataset_ood_te = create_feat_noise_dataset(data)
    

    return dataset_ind, dataset_ood_tr, dataset_ood_te


class HumLocDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(HumLocDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['edge_list.csv', 'features.csv', 'labels.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download your raw data files here
        pass

    def process(self):
        # Read CSV files with header=None and usecols=[0, 1] to select only the first and second columns
        edges_df = pd.read_csv(self.raw_paths[0], header=0, usecols=[0, 1], dtype=int)
        features_df = pd.read_csv(self.raw_paths[1], sep=',', header=None)
        labels_df = pd.read_csv(self.raw_paths[2], sep=',', header=None)

        # Build the PyG Data object
        edge_index = torch.tensor(edges_df.values, dtype=torch.long).t().contiguous()
        x = torch.tensor(features_df.values, dtype=torch.float)
        y = torch.tensor(labels_df.values, dtype=torch.float)

        # Create a PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, y=y)

        torch.save((self.collate([data])), self.processed_paths[0])

def load_humloc_dataset(data_dir, ood_type):
    dataset = HumLocDataset(root='../data/humloc')
    data = dataset[0]
    data.node_idx = torch.arange(data.num_nodes)

    dataset_ind = data
    dataset_ood_tr = data
    dataset_ood_te = data 

    if ood_type == 'feature':
        dataset_ood_tr = create_feat_noise_dataset(data)
        dataset_ood_te = create_feat_noise_dataset(data)
    

    return dataset_ind, dataset_ood_tr, dataset_ood_te

class EukLocDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(EukLocDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['edge_list.csv', 'features.csv', 'labels.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download your raw data files here
        pass

    def process(self):
        # Read CSV files with header=None and usecols=[0, 1] to select only the first and second columns
        edges_df = pd.read_csv(self.raw_paths[0], header=0, usecols=[0, 1], dtype=int)
        features_df = pd.read_csv(self.raw_paths[1], sep=',', header=None)
        labels_df = pd.read_csv(self.raw_paths[2], sep=',', header=None)

        # Build the PyG Data object
        edge_index = torch.tensor(edges_df.values, dtype=torch.long).t().contiguous()
        x = torch.tensor(features_df.values, dtype=torch.float)
        y = torch.tensor(labels_df.values, dtype=torch.float)

        # Create a PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, y=y)

        torch.save((self.collate([data])), self.processed_paths[0])


def load_eukloc_dataset(data_dir, ood_type):
    dataset = HumLocDataset(root='../data/eukloc')
    data = dataset[0]
    data.node_idx = torch.arange(data.num_nodes)

    dataset_ind = data
    dataset_ood_tr = data
    dataset_ood_te = data 

    if ood_type == 'feature':
        dataset_ood_tr = create_feat_noise_dataset(data)
        dataset_ood_te = create_feat_noise_dataset(data)
    

    return dataset_ind, dataset_ood_tr, dataset_ood_te

'''
HumLoc as in-distribution dataset,
EukLoc as out-of-distribution dataset.
'''
def load_hum_euk_dataset(data_dir):
    dataset_in = HumLocDataset(root='../data/humloc')
    d_in = dataset_in[0]
    d_in.node_idx = torch.arange(d_in.num_nodes)

    dataset_out = EukLocDataset(root='../data/eukloc')
    d_out = dataset_out[0]
    d_out.node_idx = torch.arange(d_out.num_nodes)

    dataset_ind = d_in
    dataset_ood_te = d_out
    dataset_ood_tr = d_in

    return dataset_ind, dataset_ood_tr, dataset_ood_te


def load_arxiv_dataset(data_dir, time_bound=[2015,2017], inductive=True):
    from ogb.nodeproppred import NodePropPredDataset

    ogb_dataset = NodePropPredDataset(name='ogbn-arxiv', root=f'{data_dir}/ogb')
    edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
    node_feat = torch.as_tensor(ogb_dataset.graph['node_feat'])
    label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)
    year = ogb_dataset.graph['node_year']

    year_min, year_max = time_bound[0], time_bound[1]
    test_year_bound = [2017, 2018, 2019, 2020]

    center_node_mask = (year <= year_min).squeeze(1)
    if inductive:
        ind_edge_index, _ = subgraph(center_node_mask, edge_index)
    else:
        ind_edge_index = edge_index

    dataset_ind = Data(x=node_feat, edge_index=ind_edge_index, y=label)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[center_node_mask]

    center_node_mask = (year <= year_max).squeeze(1) * (year > year_min).squeeze(1)
    if inductive:
        all_node_mask = (year <= year_max).squeeze(1)
        ood_tr_edge_index, _ = subgraph(all_node_mask, edge_index)
        
        # print("ood_tr:",year)
        # pdb.set_trace()
    else:
        ood_tr_edge_index = edge_index

    dataset_ood_tr = Data(x=node_feat, edge_index=ood_tr_edge_index, y=label)
    idx = torch.arange(label.size(0))
    dataset_ood_tr.node_idx = idx[center_node_mask]

    dataset_ood_te = []
    for i in range(len(test_year_bound)-1):
        center_node_mask = (year <= test_year_bound[i+1]).squeeze(1) * (year > test_year_bound[i]).squeeze(1)
        if inductive:
            all_node_mask = (year <= test_year_bound[i+1]).squeeze(1)
            ood_te_edge_index, _ = subgraph(all_node_mask, edge_index)
            
            # print("ood_te:", year)
            # pdb.set_trace()
        else:
            ood_te_edge_index = edge_index

        dataset = Data(x=node_feat, edge_index=ood_te_edge_index, y=label)
        idx = torch.arange(label.size(0))
        dataset.node_idx = idx[center_node_mask]
        dataset_ood_te.append(dataset)

    return dataset_ind, dataset_ood_tr, dataset_ood_te

def load_proteins_dataset(data_dir, subset_percentage, inductive=True):
    from ogb.nodeproppred import NodePropPredDataset

    ogb_dataset = NodePropPredDataset(name='ogbn-proteins', root=f'{data_dir}/ogb')

    edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
    edge_feat = torch.as_tensor(ogb_dataset.graph['edge_feat'])
    label = torch.as_tensor(ogb_dataset.labels)

    edge_index_ = to_sparse_tensor(edge_index, edge_feat, ogb_dataset.graph['num_nodes'])
    node_feat = edge_index_.mean(dim=1)

    node_species = torch.as_tensor(ogb_dataset.graph['node_species'])
    species = [0] + node_species.unique().tolist()
    ind_species_min, ind_species_max = species[0], species[3]
    ood_tr_species_min, ood_tr_species_max = species[3], species[5]
    ood_te_species = [species[i] for i in range(5, 8)]
    # ood_te_species_min, ood_te_species_max = species[5], species[8]

    center_node_mask = (node_species <= ind_species_max).squeeze(1) * (node_species > ind_species_min).squeeze(1)

    # use only a subset of the data based on the specified percentage
    num_nodes = len(center_node_mask)
    subset_size = int(subset_percentage * num_nodes)
    subset_indices = random.sample(range(num_nodes), subset_size)
    center_node_mask = torch.zeros_like(center_node_mask)
    center_node_mask[subset_indices] = True

    if inductive:
        # pdb.set_trace()
        # all_node_mask = (node_species <= ind_species_max).squeeze(1)
        # ind_edge_index, _ = subgraph(all_node_mask, edge_index)
        ind_edge_index, _ = subgraph(center_node_mask, edge_index)
    else:
        ind_edge_index = edge_index

    dataset_ind = Data(x=node_feat, edge_index=ind_edge_index, y=label)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[center_node_mask]

    # update
    center_node_mask = (node_species <= ood_tr_species_max).squeeze(1) * (node_species > ood_tr_species_min).squeeze(1)

    num_nodes = len(center_node_mask)
    subset_size = int(subset_percentage * num_nodes)
    subset_indices = random.sample(range(num_nodes), subset_size)
    center_node_mask = torch.zeros_like(center_node_mask)
    center_node_mask[subset_indices] = True

    if inductive:
        # all_node_mask = (node_species <= ood_tr_species_max).squeeze(1)
        # ood_tr_edge_index, _ = subgraph(all_node_mask, edge_index)
        ood_tr_edge_index, _ = subgraph(center_node_mask, edge_index)
    else:
        ood_tr_edge_index = edge_index

    dataset_ood_tr = Data(x=node_feat, edge_index=ood_tr_edge_index, y=label)
    idx = torch.arange(label.size(0))
    dataset_ood_tr.node_idx = idx[center_node_mask]

    dataset_ood_te = []
    for i in ood_te_species:
        center_node_mask = (node_species == i).squeeze(1)
        num_nodes = len(center_node_mask)
        subset_size = int(subset_percentage * num_nodes)
        subset_indices = random.sample(range(num_nodes), subset_size)
        center_node_mask = torch.zeros_like(center_node_mask)
        center_node_mask[subset_indices] = True
        if inductive:
            ood_te_edge_index, _ = subgraph(center_node_mask, edge_index)
        else:
            ood_te_edge_index = edge_index
        # dataset = Data(x=node_feat, edge_index=edge_index, y=label)
        dataset = Data(x=node_feat, edge_index=ood_te_edge_index, y=label)
        idx = torch.arange(label.size(0))
        dataset.node_idx = idx[center_node_mask]
        dataset_ood_te.append(dataset)

    return dataset_ind, dataset_ood_tr, dataset_ood_te



def create_sbm_dataset(data, p_ii=1.5, p_ij=0.5):
    n = data.num_nodes

    d = data.edge_index.size(1) / data.num_nodes / (data.num_nodes - 1)
    num_blocks = int(data.y.max()) + 1
    p_ii, p_ij = p_ii * d, p_ij * d
    block_size = n // num_blocks
    block_sizes = [block_size for _ in range(num_blocks-1)] + [block_size + n % block_size]
    edge_probs = torch.ones((num_blocks, num_blocks)) * p_ij
    edge_probs[torch.arange(num_blocks), torch.arange(num_blocks)] = p_ii
    edge_index = stochastic_blockmodel_graph(block_sizes, edge_probs)

    dataset = Data(x=data.x, edge_index=edge_index, y=data.y)
    dataset.node_idx = torch.arange(dataset.num_nodes)
    
    return dataset

def create_feat_noise_dataset(data):

    x = data.x
    n = data.num_nodes
    idx = torch.randint(0, n, (n, 2))
    weight = torch.rand(n).unsqueeze(1)
    x_new = x[idx[:, 0]] * weight + x[idx[:, 1]] * (1 - weight)

    dataset = Data(x=x_new, edge_index=data.edge_index, y=data.y)
    dataset.node_idx = torch.arange(n)

    return dataset

def create_label_noise_dataset(data):

    y = data.y
    n = data.num_nodes
    idx = torch.randperm(n)[:int(n * 0.5)]
    y_new = y.clone()
    y_new[idx] = torch.randint(0, y.max(), (int(n * 0.5), ))

    dataset = Data(x=data.x, edge_index=data.edge_index, y=y_new)
    dataset.node_idx = torch.arange(n)

    return dataset

def load_graph_dataset(data_dir, dataname, ood_type):
    transform = T.NormalizeFeatures()
    if dataname in ('cora', 'citeseer', 'pubmed'):
        torch_dataset = Planetoid(root=f'{data_dir}Planetoid', split='public',
                              name=dataname, transform=transform)
        dataset = torch_dataset[0]
        tensor_split_idx = {}
        idx = torch.arange(dataset.num_nodes)
        tensor_split_idx['train'] = idx[dataset.train_mask]
        tensor_split_idx['valid'] = idx[dataset.val_mask]
        tensor_split_idx['test'] = idx[dataset.test_mask]
        dataset.splits = tensor_split_idx
    elif dataname == 'amazon-photo':
        torch_dataset = Amazon(root=f'{data_dir}Amazon',
                               name='Photo', transform=transform)
        dataset = torch_dataset[0]
    elif dataname == 'amazon-computer':
        torch_dataset = Amazon(root=f'{data_dir}Amazon',
                               name='Computers', transform=transform)
        dataset = torch_dataset[0]
    elif dataname == 'coauthor-cs':
        torch_dataset = Coauthor(root=f'{data_dir}Coauthor',
                                 name='CS', transform=transform)
        dataset = torch_dataset[0]
    elif dataname == 'coauthor-physics':
        torch_dataset = Coauthor(root=f'{data_dir}Coauthor',
                                 name='Physics', transform=transform)
        dataset = torch_dataset[0]
    else:
        raise NotImplementedError

    dataset.node_idx = torch.arange(dataset.num_nodes)
    dataset_ind = dataset

    if ood_type == 'structure':
        dataset_ood_tr = create_sbm_dataset(dataset, p_ii=1.5, p_ij=0.5)
        dataset_ood_te = create_sbm_dataset(dataset, p_ii=1.5, p_ij=0.5)
    elif ood_type == 'feature':
        dataset_ood_tr = create_feat_noise_dataset(dataset)
        dataset_ood_te = create_feat_noise_dataset(dataset)
    elif ood_type == 'label':
        if dataname == 'cora':
            class_t = 3
        elif dataname == 'amazon-photo':
            class_t = 4
        elif dataname == 'coauthor-cs':
            class_t = 4
        label = dataset.y

        center_node_mask_ind = (label > class_t)
        idx = torch.arange(label.size(0))
        dataset_ind.node_idx = idx[center_node_mask_ind]

        if dataname in ('cora', 'citeseer', 'pubmed'):
            split_idx = dataset.splits
        if dataname in ('cora', 'citeseer', 'pubmed', 'arxiv'):
            tensor_split_idx = {}
            idx = torch.arange(label.size(0))
            for key in split_idx:
                mask = torch.zeros(label.size(0), dtype=torch.bool)
                mask[torch.as_tensor(split_idx[key])] = True
                tensor_split_idx[key] = idx[mask * center_node_mask_ind]
            dataset_ind.splits = tensor_split_idx

        dataset_ood_tr = Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y)
        dataset_ood_te = Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y)

        center_node_mask_ood_tr = (label == class_t)
        center_node_mask_ood_te = (label < class_t)
        dataset_ood_tr.node_idx = idx[center_node_mask_ood_tr]
        dataset_ood_te.node_idx = idx[center_node_mask_ood_te]
    else:
        raise NotImplementedError

    return dataset_ind, dataset_ood_tr, dataset_ood_te
