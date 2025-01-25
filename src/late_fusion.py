import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from src.utils import _train, _validate



def get_predictions(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())  # Zapisujemy pełne wyniki (np. logits lub softmax)
            true_labels.extend(labels.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)  # Łączymy wyniki dla całego zbioru
    return predictions, true_labels



def train_late_fusion(models: list[torch.nn.Module], 
                      optimizers: list[torch.optim.Optimizer], 
                      train_loaders: list[torch.utils.data.DataLoader], 
                      test_loaders: list[torch.utils.data.DataLoader], 
                      criterion: torch.nn.Module, 
                      num_epochs: int, 
                      device: torch.device):
    
    losses = []
    accs = []
    precs = []
    recs = []
    max_epoch = 0
    best_acc = 0
    best_epoch = 0
    best_cm = None
    
    for epoch in range(num_epochs):
        all_predictions = []
        labels = None
        val_losses = []
        val_accs = []
        val_precs = []
        val_recs = []
        for i, model in enumerate(models):
            _, _, _, _, _ = _train(model, train_loaders[i], criterion, optimizers[i], device)
            val_loss, val_acc, val_prec, val_rec, _ = _validate(model, test_loaders[i], criterion, device)
            predictions, labels = get_predictions(model, test_loaders[i], device)
            all_predictions.append(predictions)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            val_precs.append(val_prec)
            val_recs.append(val_rec)
            
        all_predictions = np.array(all_predictions)
        # final_predictions = np.mean(all_predictions, axis = 0) 
        num_models, num_samples, num_classes = all_predictions.shape
        stacked_predictions = all_predictions.transpose(1, 0, 2).reshape(num_samples, -1)
        labels = np.array(labels)
        meta_learner = LogisticRegression(max_iter = 1000)
        meta_learner.fit(stacked_predictions, labels)
        final_predictions = meta_learner.predict_proba(stacked_predictions)
        final_labels = np.argmax(final_predictions, axis = 1)
        
        final_predictions_tensor = torch.tensor(final_predictions, dtype=torch.float32).to(device)
        labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
        fusion_loss = criterion(final_predictions_tensor, labels_tensor).item()
    
        accuracy = accuracy_score(labels, final_labels)
        precision = precision_score(labels, final_labels, average = "weighted", zero_division = 0)
        recall = recall_score(labels, final_labels, average = "weighted", zero_division = 0)
        cm = confusion_matrix(labels, final_labels)
        
        losses.append(fusion_loss)
        accs.append(accuracy)
        precs.append(precision)
        recs.append(recall)
        max_epoch += 1
        
        if val_acc > best_acc:
            best_acc = accuracy
            best_epoch = epoch
            best_cm = cm
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Val Loss Audio: {val_losses[0]:.4f},\
                    Val Accuracy Audio: {val_accs[0]:.4f},\
                    Val Precision Audio: {val_precs[0]:.4f},\
                    Val Recall Audio: {val_recs[0]:.4f}.")
            print(f"Val Loss Video: {val_losses[1]:.4f},\
                    Val Accuracy Video: {val_accs[1]:.4f},\
                    Val Precision Video: {val_precs[1]:.4f},\
                    Val Recall Video: {val_recs[1]:.4f}.")
            print(f"Fusion Loss: {fusion_loss:.4f},\
                    Fusion Accuracy: {accuracy:.4f},\
                    Fusion Precision: {precision:.4f},\
                    Fusion Recall: {recall:.4f}.")
            print("-" * 50)
            
    print(f"BEST EPOCH: {best_epoch:.2f}\
          BEST LOSS: {losses[best_epoch]:.2f}\
          BEST ACCURACY: {accs[best_epoch]:.2f}\
          BEST PRECISION: {precs[best_epoch]:.2f}\
          BEST RECALL: {recs[best_epoch]:.2f}")
        
    return [*range(1, max_epoch + 1)], losses, accs, precs, recs, best_epoch, best_cm
        
        