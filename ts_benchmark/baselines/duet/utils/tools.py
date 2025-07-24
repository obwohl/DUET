import copy
import os

import numpy as np
import math
import torch
import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args, verbose=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.lr * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }

    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.lr if epoch < 3 else args.lr * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.lr}
    elif args.lradj == 'cosine_warmup':
        warmup_epochs = getattr(args, 'warmup_epochs', 10)
        if epoch <= warmup_epochs:
            # linear warmup
            lr = args.lr * (epoch / warmup_epochs)
        else:
            # cosine annealing
            progress = (epoch - warmup_epochs) / (args.num_epochs - warmup_epochs)
            lr = 0.5 * args.lr * (1. + math.cos(math.pi * progress))
        lr_adjust = {epoch: lr}
    elif args.lradj == 'plateau':
        return
    else:
        lr_adjust = {}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if verbose:
            print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.check_point = None


    def __call__(self, val_loss, model_state):
        # --- START OF FIX: Robust handling of NaN/Inf ---
        # 1. Check for invalid loss values (NaN or Inf).
        #    These are always treated as a non-improvement.
        if np.isnan(val_loss) or np.isinf(val_loss):
            self.counter += 1
            if self.verbose:
                # Print a clear message why the counter is being incremented.
                print(f"Invalid validation loss ({val_loss}). EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return  # Exit the function here to avoid the faulty logic below.

        # 2. The remaining logic is only executed for valid, finite loss values.
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model_state)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model_state)
            self.counter = 0
        # --- END OF FIX ---

    def save_checkpoint(self, val_loss, model_state):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        # --- DIE KORREKTUR ---
        # VORHER (falsch): self.check_point = copy.deepcopy(model.state_dict())
        # NACHHER (korrekt): Die Variable `model_state` IST bereits das state_dict.
        # Wir müssen es nur noch kopieren.
        self.check_point = copy.deepcopy(model_state)
        # --- ENDE KORREKTUR ---
        
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
