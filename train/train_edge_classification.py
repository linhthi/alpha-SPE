import torch
import torch.nn as nn
import math
import dgl
from train.metrics import accuracy_SBM as accuracy
from tqdm import tqdm
from torch.utils.data import DataLoader

def train_epoch(model, optimizer, device, graph, epoch, batch_size):

    model.train()
    epoch_loss = 0
    epoch_train_accuracy = 0
   
    for perm in tqdm(DataLoader(range(graph.edges().size()[0]), batch_size, shuffle=True)):
        optimizer.zero_grad()
        graph = graph.to(device)
        edges = graph.edges()[0][perm].to(device), graph.edges()[1][perm].to(device)
        e = graph.edata['h'][perm].to(device)
        h = model(graph)
        pos_out = model.edge_predictor(h[edges[0]], h[edges[1]])

        egde_neg = torch.rand(0, h.size(0), edges[0].size(0), dtype=torch.long, device=device)
        neg_out = model.edge_predictor(h[egde_neg[0]], h[egde_neg[1]])

        loss = model.loss(pos_out, neg_out)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()
    
    return epoch_loss, epoch_train_accuracy, optimizer

def evaluate_network(model, device, graph, batch_size, epoch):
    
    model.eval()
    epoch_test_loss = 0
    epoch_test_accuracy = 0
    with torch.no_grad():
        for perm in tqdm(DataLoader(range(graph.edges().size()[0]), batch_size, shuffle=True)):
            graph = graph.to(device)
            edges = graph.edges()[0][perm].to(device), graph.edges()[1][perm].to(device)
            e = graph.edata['h'][perm].to(device)
            h = model(graph)
            pos_out = model.edge_predictor(h[edges[0]], h[edges[1]])

            egde_neg = torch.rand(0, h.size(0), edges[0].size(0), dtype=torch.long, device=device)
            neg_out = model.edge_predictor(h[egde_neg[0]], h[egde_neg[1]])

            loss = model.loss(pos_out, neg_out)
            acc = accuracy(pos_out, e)
            epoch_loss += loss.detach().item()
    
    return epoch_loss, acc
        
    return epoch_test_loss, epoch_test_accuracy