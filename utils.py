import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, f1_score





def print_model_metrics(y_train, y_val, y_train_preds, y_val_preds, y_train_probs, y_val_probs, metrics="all", tofile=None):
    '''print model metrics to stdout or a file.'''
    # accuracy
    if metrics == "all" or "acc" in metrics:
        train_acc = accuracy_score(y_train, y_train_preds)
        val_acc = accuracy_score()