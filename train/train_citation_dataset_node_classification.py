"""
    Utility function for training one epoch
    and evaluating one epoch on citation graph: Cora, Pubmed, Citeseer.
    These dataset contain only node feartures.
"""
import torch
import torch.nn as nn
import math
import dgl

from train.metrics import accuracy_SBM as accuracy


def train_epoch(model, optimizer, device, graph, mask):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0

    graph = graph.to(device)
    feats = graph.ndata['feat'].to(device)
    # print(graph.nodes().size(), feats.size())

    labels = graph.ndata['label'].to(device)
    optimizer.zero_grad()
    # print(labels.size())
    preds = model.forward(graph, feats)

    loss = model.loss(preds[mask], labels[mask])
    loss.backward()
    optimizer.step()
    epoch_loss = loss.detach().item()
    epoch_train_acc = accuracy(preds, labels)

    return epoch_loss, epoch_train_acc, optimizer


def evaluate_network(model, device, graph, mask):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0

    with torch.no_grad():
        graph = graph.to(device)
        feats = graph.ndata['feat'].to(device)
        labels = graph.ndata['label'].to(device)
        preds = model.forward(graph, feats)
        loss = model.loss(preds[mask], labels[mask])
        epoch_test_loss = loss.detach().item()
        epoch_test_acc = accuracy(preds, labels)

    return epoch_test_loss, epoch_test_acc

