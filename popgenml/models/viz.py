# -*- coding: utf-8 -*-
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns

def count_parameters(model, f = None):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table, file = f)
    print(f"Total Trainable Params: {total_params}", file = f)
    return total_params

"""
Plots piecewise constant size history. Must call plt.show() or plt.savefig() after one more calls.
"""
def plot_size_history(Nt, max_t = None, color = 'k'):
    N = [u[0] for u in Nt]
    t = [u[1] for u in Nt]

    N = N[1:]
    t = np.log10(t[1:])

    for k in range(len(N) - 1):
        plt.plot([t[k], t[k + 1]], [N[k], N[k]], c = color)
        plt.plot([t[k + 1], t[k + 1]], [N[k], N[k + 1]], c = color)

    if max_t:
        plt.plot([t[-1], max_t], [N[-1], N[-1]], c = color)

def plot_confusion_matrix(y_true, y_pred, labels, filename = None, ymap=None, figsize=(10,10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred)
    
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    
    if filename is not None:
        plt.savefig(filename, dpi = 100)
    else:
        plt.show()
    plt.close()