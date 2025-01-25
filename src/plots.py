import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
from src.extra import load_yaml


def plot_metrics(train_losses, val_losses, train_accs, val_accs, train_precs, val_precs, train_recs, val_recs):
    epochs = [*range(1, len(train_losses) + 1)]
    combinations = list(itertools.product([0, 1], repeat = 2))
    titles = ['Loss', 'Accuracy', 'Precision', 'Recall']
    
    _, axs = plt.subplots(2, 2, figsize = (15, 10))
    axs[0][0].plot(epochs, train_losses, color = 'blue', linestyle = 'dashdot', label = 'train loss')
    axs[0][0].plot(epochs, val_losses, color = 'blue', linestyle = 'solid', label = 'val loss')
    axs[0][1].plot(epochs, train_accs, color = 'red', linestyle = 'dashdot', label = 'train acc')
    axs[0][1].plot(epochs, val_accs, color = 'red', linestyle = 'solid', label = 'val acc')
    axs[1][0].plot(epochs, train_precs, color = 'green', linestyle = 'dashdot', label = 'train loss')
    axs[1][0].plot(epochs, val_precs, color = 'green', linestyle = 'solid', label = 'val loss')
    axs[1][1].plot(epochs, train_recs, color = 'gold', linestyle = 'dashdot', label = 'train acc')
    axs[1][1].plot(epochs, val_recs, color = 'gold', linestyle = 'solid', label = 'val acc')
    
    for idx, (i, j) in enumerate(combinations):
        axs[i][j].set_title(titles[idx], fontsize = 10)
        axs[i][j].set_xlabel('Epoch')
        axs[i][j].set_ylabel(titles[idx])
        
    plt.legend()
    plt.suptitle('Model results', fontsize = 20)
    plt.tight_layout()
    plt.show()



def plot_metrics_late_fusion(losses, accs, precs, recs):
    epochs = [*range(1, len(losses) + 1)]
    combinations = list(itertools.product([0, 1], repeat = 2))
    titles = ['Loss', 'Accuracy', 'Precision', 'Recall']
    
    _, axs = plt.subplots(2, 2, figsize = (15, 10))
    axs[0][0].plot(epochs, losses, color = 'blue', linestyle = 'losses', label = 'fusion loss')
    axs[0][1].plot(epochs, accs, color = 'red', linestyle = 'losses', label = 'fusion acc')
    axs[1][0].plot(epochs, precs, color = 'green', linestyle = 'losses', label = 'fusion prec')
    axs[1][1].plot(epochs, recs, color = 'gold', linestyle = 'losses', label = 'train rec')
    
    
    for idx, (i, j) in enumerate(combinations):
        axs[i][j].set_title(titles[idx], fontsize = 10)
        axs[i][j].set_xlabel('Epoch')
        axs[i][j].set_ylabel(titles[idx])
        axs[i][j].legend()
        
    plt.suptitle('Model results', fontsize = 20)
    plt.tight_layout()
    plt.show()



def plot_confusion_matrix(cm, normalize = True, cmap = 'Greens'):
    plt.figure(figsize = (10, 8))
    vars = load_yaml('../params.yaml')
    kws = vars['emotion_mapping']
    label_map = {int(v): k for k, v in kws.items()}
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        
    sns.heatmap(cm, 
                annot = True, 
                fmt = '.2f' if normalize else 'd',
                cmap = cmap, 
                xticklabels = [label_map[i] for i in range(cm.shape[1])], 
                yticklabels  =[label_map[i] for i in range(cm.shape[0])])
    plt.title('Model heatmap', fontsize = 20)
    plt.show()



def plot_multiple_confusion_matrices(**cmaps):
    
    _, axs = plt.subplots(1, len(cmaps), figsize = (12, 6))
    vars = load_yaml('../params.yaml')
    kws = vars['emotion_mapping']
    label_map = {int(v): k for k, v in kws.items()}
    
    for i, (title, cm) in enumerate(cmaps.items()):
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]    
        sns.heatmap(cm, 
                    annot = True, 
                    fmt = '.2f',
                    cmap = 'Greens', 
                    xticklabels = [label_map[i] for i in range(cm.shape[1])], 
                    yticklabels = [label_map[i] for i in range(cm.shape[0])],
                    ax = axs[i])
        axs[i].set_title(title, fontsize = 15)
    plt.tight_layout()
    plt.show()