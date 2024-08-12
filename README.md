# ML-GOOD
## Multi-Label Datasets
- PPI(Protein-Protein Interaction)
In general, if two proteins are involved in a life process or perform a function together, it is considered that there is an interaction between the two proteins.
The gene ontology base acts as a label(121 in total), and labels are not one-hot encoding.
> Zitnik, M.; and Leskovec, J. 2017. Predicting multicellular function through multi-layer tissue networks. Bioinformatics, (14): i190–i198.

How To Load👇
```python
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
            # dataset_ind = dataset
            dataset_ind.append(dataset)
        elif i < valid_idx:
            dataset_ood_tr.append(dataset)
        else:
            dataset_ood_te.append(dataset)

    return dataset_ind, dataset_ood_tr, dataset_ood_te
```
- DBLP
There are four types of nodes in the DBLP data set: author, paper, term, and conference.
We use its multi-label forms from $MLGNC^{[1]}$.
> [1] Zhao, T.; Dong, T. N.; Hanjalic, A.; and Khosla, M. 2023. Multi-label Node Classification On Graph-Structured Data. Transactions on Machine Learning Research. URL https://openreview.net/forum?id=EZhkV2BjDP.

How To Load👇
```python
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
        with open(self.raw_paths[1], 'r') as file:
            edges = [list(map(int, line.strip().split())) for line in file]

        with open(self.raw_paths[2], 'r') as file:
            labels = [list(map(float, line.strip().split(','))) for line in file]

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
```

- PCG
How To Load👇
```python
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
        x = torch.tensor(features_df.values, dtype=torch.float)
        y = torch.tensor(labels_df.values, dtype=torch.float)

        # Create a PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, y=y)

        torch.save((self.collate([data])), self.processed_paths[0])


def load_dblp_dataset(data_dir, ood_type):
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
```

- HumLoc
  How to Load👇
```python
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
```
- EukLoc
  How to Load👇
```python
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
```

- OGBN-Proteins
> Hu, W.; Fey, M.; Zitnik, M.; Dong, Y.; Ren, H.; Liu, B.; Catasta, M.; and Leskovec, J. 2020. Open Graph Benchmark: Datasets for Machine Learning on Graphs. In NeurIPS, 22118–22133.
  
 How to Load👇
  ```python
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
  ```
  ## Run
  cross-dataset
  ```bash
  python main.py --backbone gcn --method mlgood --dataset ppipcg --use_reg --m_in -9 --m_out -4 --lamda 0.01 --use_emo
  ```
