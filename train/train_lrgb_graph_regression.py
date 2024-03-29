"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import numpy as np

from train.metrics import MAE

"""
    For GCNs
"""
def train_epoch(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_targets = batch_targets.to(device)
        optimizer.zero_grad()
        try:
            sign_flip = torch.rand(batch_pos_enc.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            batch_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
            batch_scores = model.forward(batch_graphs, batch_x, batch_e)
        except:
            batch_scores = model.forward(batch_graphs, batch_x, batch_e)
        loss = model.loss(batch_scores, batch_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_mae += MAE(batch_scores, batch_targets)
        nb_data += batch_targets.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)
    
    return epoch_loss, epoch_train_mae, optimizer

def evaluate_network(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    nb_data = 0
    with torch.no_grad():
        count = 0
        for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_targets = batch_targets.to(device)
            try:
                batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            except:
                batch_scores = model.forward(batch_graphs, batch_x, batch_e)

            #print(batch_scores.cpu().detach().numpy())
            # print(batch_targets.cpu().detach().numpy())

            loss = model.loss(batch_scores, batch_targets)
            if np.isnan(loss.detach().item()):
                #print("output: ", batch_scores.cpu().detach().numpy())
                #print("target: ", batch_targets.cpu().detach().numpy())
                #print(batch_x.cpu().detach().numpy())
                #print(batch_e.cpu().detach().numpy())
                output = batch_scores.cpu().detach().numpy()
                sum_o = np.sum(output, axis=1)
                for i in range(len(sum_o)):
                    if np.isnan(sum_o[i]):
                        print(output[i])
                        print(32*count + 1)
                        print(batch_graphs.ndata["SE"][i].cpu().detach().numpy())
                        print(batch_graphs.ndata["EigVals"][i].cpu().detach().numpy())
                        print(batch_graphs.ndata["EigVecs"][i].cpu().detach().numpy())
                
            count += 1

            epoch_test_loss += loss.detach().item()
            epoch_test_mae += MAE(batch_scores, batch_targets)
            nb_data += batch_targets.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)
        
    return epoch_test_loss, epoch_test_mae




