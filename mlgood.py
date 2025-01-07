import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree
from backbone import *
import numpy as np
import pdb

class MLGOOD(nn.Module):
    '''
    Energy-based out-of-distribution detection model class 
    for multi-label node classification.
    '''
    def __init__(self, d, c, args):
        super(MLGOOD, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=args.use_bn)
        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                        out_channels=c, num_layers=args.num_layers,
                        dropout=args.dropout)
        elif args.backbone == 'gat':
            self.encoder = GAT(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout, use_bn=args.use_bn)
        elif args.backbone == 'mixhop':
            self.encoder = MixHop(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout)
        else:
            raise NotImplementedError

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        '''return predicted logits'''
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index)

    def propagation(self, e, edge_index, prop_layers=1, alpha=0.5):
        '''energy belief propagation, return the energy after propagation'''
        e = e.unsqueeze(1)
        N = e.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm = 1. / d[col]
        value = torch.ones_like(row) * d_norm
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        for _ in range(prop_layers):
            e = e * alpha + matmul(adj, e) * (1 - alpha)
        return e.squeeze(1)

    def detect(self, dataset, node_idx, device, args):
        '''return negative energy, a vector for all input nodes'''
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logits = self.encoder(x, edge_index)
        if args.dataset in ('proteins', 'ppi', 'dblp', 'pcg', 'humloc', 'eukloc', 'humeuk', 'ppipcg'): # for multi-label binary classification
            logits = torch.stack([logits, torch.zeros_like(logits)], dim=2)
            if args.use_emo:
                emo_energy_sum = args.T * torch.logsumexp(logits / args.T, dim=-1).sum(dim=1)
                log_sum_exp_result = torch.logsumexp(logits / args.T, dim=-1)
                emo_energy = args.T * log_sum_exp_result.mean(dim=1)
                emo_energy += emo_energy_sum
                neg_energy = emo_energy
            else:
                neg_energy = args.T * torch.max(torch.logsumexp(logits / args.T, dim=-1), dim=1).values
        else: # for single-label multi-class classification
            neg_energy = args.T * torch.logsumexp(logits / args.T, dim=-1)
        if args.use_prop: # use energy belief propagation
            neg_energy = self.propagation(neg_energy, edge_index, args.K, args.alpha)
        return neg_energy[node_idx]

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):
        '''return loss for training'''
        if isinstance(dataset_ind, list):
            loss = .0
            for i, di in enumerate(dataset_ind):
                if isinstance(dataset_ood, list):
                    do = dataset_ood[i]
                    x_in, edge_index_in = di.x.to(device), di.edge_index.to(device)
                    x_out, edge_index_out = do.x.to(device), do.edge_index.to(device)
                    logits_in = self.encoder(x_in, edge_index_in)
                    logits_out = self.encoder(x_out, edge_index_out)
                    train_in_idx, train_ood_idx = di.split['train'], do.node_idx
                    if args.dataset in ('proteins', 'ppi', 'dblp', 'pcg', 'humloc', 'eukloc', 'humeuk', 'ppipcg'):
                        sup_loss = criterion(logits_in[train_in_idx], di.y[train_in_idx].to(device).to(torch.float))
                    else:
                        pred_in = F.log_softmax(logits_in[train_in_idx], dim=1)
                        sup_loss = criterion(pred_in, di.y[train_in_idx].squeeze(1).to(device))
                    if args.use_reg:
                        if args.dataset in ('proteins', 'ppi', 'dblp', 'pcg', 'humloc', 'eukloc', 'humeuk', 'ppipcg'):
                            logits_in = torch.stack([logits_in, torch.zeros_like(logits_in)], dim=2)
                            logits_out = torch.stack([logits_out, torch.zeros_like(logits_out)], dim=2)
                            if args.use_emo:
                                log_sum_exp_result_in = torch.logsumexp(logits_in / args.T, dim=-1)
                                emo_energy_in_mean = -args.T * log_sum_exp_result_in.mean(dim=1)
                                emo_energy_in_sum = -args.T * log_sum_exp_result_in.sum(dim=1)
                                energy_in = emo_energy_in_mean + emo_energy_in_sum
                                log_sum_exp_result_out = torch.logsumexp(logits_out / args.T, dim=-1)
                                emo_energy_out_mean = -args.T * log_sum_exp_result_out.mean(dim=1)
                                emo_energy_out_sum = -args.T * log_sum_exp_result_out.sum(dim=1)
                                energy_out = emo_energy_out_mean + emo_energy_out_sum
                            else:
                                energy_in = - args.T * torch.max(torch.logsumexp(logits_in / args.T, dim=-1), dim=1).values
                                energy_out = - args.T * torch.max(torch.logsumexp(logits_out / args.T, dim=-1), dim=1).values
                        else:
                            energy_in = - args.T * torch.logsumexp(logits_in / args.T, dim=-1)
                            energy_out = - args.T * torch.logsumexp(logits_out / args.T, dim=-1)
                        if args.use_prop:
                            energy_in = self.propagation(energy_in, edge_index_in, args.K, args.alpha)[train_in_idx]
                            energy_out = self.propagation(energy_out, edge_index_out, args.K, args.alpha)[train_ood_idx]
                        else:
                            energy_in = energy_in[train_in_idx]
                            energy_out = energy_out[train_ood_idx]
                        if energy_in.shape[0] != energy_out.shape[0]:
                            min_n = min(energy_in.shape[0], energy_out.shape[0])
                            energy_in = energy_in[:min_n]
                            energy_out = energy_out[:min_n]
                        reg_loss = torch.mean(F.relu(energy_in - args.m_in) ** 2 + F.relu(args.m_out - energy_out) ** 2)
                        loss += sup_loss + args.lamda * reg_loss
                    else:
                        loss += sup_loss
        return loss
