import torch
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt




class CremaDataset(Dataset):
    
    def __init__(self, embeddings: dict[torch.tensor, torch.tensor]) -> None:
        self.embeddings = list(embeddings.keys())
        self.labels = list(embeddings.values())
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> tuple[torch.tensor, torch.tensor]:
        emb = self.embeddings[idx]
        label = self.labels[idx]
        return emb, label



def train_test_dataloader(embeddings_dict: dict, *, batch_size: int = 8, test_ratio: float = 0.2) -> tuple[DataLoader]:
    dataset = CremaDataset(embeddings_dict)
    train_size = int((1 - test_ratio) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 0)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 0)
    return train_loader, test_loader



class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int = 6, dropout = 0.3):
        super(MLP, self).__init__()
        
        layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU(), nn.Dropout(dropout)]
        
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            block = [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            layers.extend(block)
            
        last_fc = nn.Linear(hidden_dims[-1], output_dim)
        layers.append(last_fc)      
        self.model = nn.Sequential(*layers)
    
    def forward(self, input_values):
        logits = self.model(input_values)
        return logits



def _train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for inputs, labels in tqdm(dataloader, desc = "Training", leave = False):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(torch.argmax(outputs, dim = 1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average = "weighted", zero_division = 0)
    recall = recall_score(all_labels, all_preds, average = "weighted", zero_division = 0)
    cm = confusion_matrix(all_labels, all_preds)
    return total_loss / len(dataloader), accuracy, precision, recall, cm



def _validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc = "Validation", leave = False):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.extend(torch.argmax(outputs, dim = 1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average = "weighted", zero_division = 0)
    recall = recall_score(all_labels, all_preds, average = "weighted", zero_division = 0)
    cm = confusion_matrix(all_labels, all_preds)
    
    return total_loss / len(dataloader), accuracy, precision, recall, cm



def _validate_late_fusion(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_outputs.append(outputs)

    accuracy = accuracy_score(all_labels, all_preds)
    
    return total_loss / len(dataloader), accuracy, all_outputs, all_labels



def train_model(model, criterion, optimizer, num_epochs, train_loader, test_loader, device, patience = np.inf):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_precs = []
    val_precs = []
    train_recs = []
    val_recs = []
    best_acc = 0
    best_epoch = 0
    max_epoch = 0
    is_quit = 0
    cm = None
    
    for epoch in range(num_epochs):
        train_loss, train_acc, train_prec, train_rec, train_cm = _train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_prec, val_rec, val_cm = _validate(model, test_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_precs.append(train_prec)
        val_precs.append(val_prec)
        train_recs.append(train_rec)
        val_recs.append(val_rec)
        max_epoch += 1
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            cm = val_cm
            is_quit = 0
        else:
            is_quit += 1
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f},\
                    Train Accuracy: {train_acc:.4f},\
                    Train Precision: {train_prec:.4f},\
                    Train Recall {train_rec:.4f}.")
            print(f"Val Loss: {val_loss:.4f},\
                    Val Accuracy: {val_acc:.4f},\
                    Val Precision: {val_prec:.4f},\
                    Val Recall: {val_rec:.4f}.")
            print("-" * 50)
        
        if is_quit > patience:
            break
    print(f"BEST EPOCH: {best_epoch:.2f}\
          BEST ACCURACY: {val_accs[best_epoch]:.2f}\
          BEST PRECISION: {val_precs[best_epoch]:.2f}\
          BEST RECALL: {val_recs[best_epoch]:.2f}")
    
    return [*range(1, max_epoch + 1)], train_losses, val_losses, train_accs, val_accs, train_precs, val_precs, train_recs, val_recs, best_epoch, cm 

        

def train_late_fusion_model(model, criterion, optimizer, num_epochs, train_loader, test_loader, device):
    all_outputs = []
    all_labels = []
    for _ in range(num_epochs):
        _, _ = _train(model, train_loader, criterion, optimizer, device)
        _, _, outputs, labels = _validate_late_fusion(model, test_loader, criterion, device)
        all_outputs.append(outputs)
        all_labels.append(labels)
       
    return [torch.cat(all_outputs[i], dim = 0) for i in range(len(all_outputs))], all_labels



def concat_embs(embs_dict1, embs_dict2):
    get_embeddings = lambda embs_dict: torch.stack(list(embs_dict.keys()), dim = 0)
    embs1 = get_embeddings(embs_dict1)
    embs2 = get_embeddings(embs_dict2)
    labels = list(embs_dict1.values())
    fused_embs = torch.cat((embs1, embs2), dim = 1)
    fused_embs = list(torch.chunk(fused_embs, chunks = len(labels), dim = 0))
    fused_embs = [emb.squeeze(0) for emb in fused_embs]
    result_dict = dict(zip(fused_embs, labels))
    return result_dict