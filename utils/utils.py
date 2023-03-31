from __future__ import division

import torch
import torch.nn as nn
from contextlib import contextmanager
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    cohen_kappa_score,
    precision_recall_curve,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    confusion_matrix
)
from collections import OrderedDict, defaultdict
from itertools import repeat
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from scipy.sparse import linalg
import sklearn
import matplotlib.cm as cm
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
import math
import tqdm
import shutil
import queue
import random
import time
import json
import torch
import h5py
import logging
import numpy as np
import os
import sys
import pickle
import scipy.sparse as sp
from collections import Counter

import warnings
warnings.filterwarnings("ignore")

MASK = 0.0
LARGE_NUM = 1e9


@contextmanager
def timer(name="Main", logger=None):
    t0 = time.time()
    yield
    msg = f"[{name}] done in {time.time() - t0} s"
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


def seed_torch(seed=123):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_save_dir(base_dir, training, id_max=500):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).
    Args:
        base_dir (str): Base directory in which to make save directories.
        training (bool): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.
    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        subdir = "train" if training else "test"
        save_dir = os.path.join(base_dir, subdir, "{}-{:02d}".format(subdir, uid))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir

    raise RuntimeError(
        "Too many save directories created with the same name. \
                       Delete old save directories or use another name."
    )

def eval_dict(
    y_pred,
    y,
    y_prob=None,
    average="binary",
    metrics=[],
    null_val=0.0,
    thresholds=None,
    find_threshold_on=None,
):
    """
    Args:
        y_pred: Predicted labels of all samples
        y : True labels of all samples
        file_names: File names of all samples
        average: 'weighted', 'micro', 'macro' etc. to compute F1 score etc.
    Returns:
        scores_dict: Dictionary containing scores such as F1, acc etc.
        pred_dict: Dictionary containing predictions
        true_dict: Dictionary containing labels
    """
    scores_dict = {}

    if find_threshold_on is not None:
        if find_threshold_on == "gbeta":
            thresholds = find_optimal_cutoff_thresholds_for_beta(y, y_prob, threshold_on="gbeta")
        elif find_threshold_on == "fbeta":
            thresholds = find_optimal_cutoff_thresholds_for_beta(y, y_prob, threshold_on="fbeta")
        elif find_threshold_on == "F1":
            thresholds = find_optimal_cutoff_thresholds_for_f1(y, y_prob)
        elif find_threshold_on == "youden":
            thresholds = find_optimal_cutoff_thresholds_youden(y, y_prob)
        else:
            raise NotImplementedError

    if thresholds is not None:
        y_pred = apply_thresholds(y_prob, thresholds)

    if "acc" in metrics:
        scores_dict["acc"] = accuracy_score(y_true=y, y_pred=y_pred)
    if "F1" in metrics:
        scores_dict["F1"] = f1_score(y_true=y, y_pred=y_pred, average=average)
    if "precision" in metrics:
        scores_dict["precision"] = precision_score(
            y_true=y, y_pred=y_pred, average=average
        )
    if "recall" in metrics:
        scores_dict["recall"] = recall_score(y_true=y, y_pred=y_pred, average=average)
    if "auroc" in metrics:
        assert y_prob is not None
        scores_dict["auroc"] = roc_auc_score(
            y_true=y, y_score=y_prob, average=average if (average!="binary") else None,
        )
    if "aupr" in metrics:
        assert y_prob is not None
        scores_dict["aupr"] = average_precision_score(
            y_true=y, 
            y_score=y_prob, 
            average=average if (average!="binary") else None,
        )
    if "specificity" in metrics:
        tn, fp, fn, tp = confusion_matrix(y_true=y, y_pred=y_pred).ravel()
        scores_dict["specificity"] = tn / (tn + fp)
    if "kappa" in metrics:
        scores_dict["kappa"] = cohen_kappa_score(y1=y, y2=y_pred)
    if "mae" in metrics:
        scores_dict["mae"] = masked_mae_np(preds=y_pred, labels=y, null_val=null_val)
    if "rmse" in metrics:
        scores_dict["rmse"] = masked_rmse_np(preds=y_pred, labels=y, null_val=null_val)
    if "mse" in metrics:
        scores_dict["mse"] = masked_mse_np(preds=y_pred, labels=y, null_val=null_val)
    if "mape" in metrics:
        scores_dict["mape"] = masked_mape_np(preds=y_pred, labels=y, null_val=null_val)
    if ("fbeta" in metrics) or ("gbeta" in metrics):
        fbeta, gbeta = compute_fbeta_gbeta(
            y_true=y, y_pred=y_pred
        )
        if average == "macro":
            scores_dict["fbeta"] = np.mean(fbeta)
            scores_dict["gbeta"] = np.mean(gbeta)
        elif (average is None): # individual classes
            scores_dict["fbeta"] = fbeta
            scores_dict["gbeta"] = gbeta
        else:
            raise NotImplementedError

    return scores_dict, thresholds


def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype("float32")
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype("float32")
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)


def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype("float32")
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype("float32")
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype("float32")
        mask /= np.mean(mask)
        mape = np.abs(
            np.divide(
                np.subtract(preds, labels).astype("float32"),
                labels,
                where=(labels != 0),
                out=np.zeros_like(labels),
            )
        )
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)

def masked_mae_loss(y_pred, y_true, null_val=0.0):
    """
    Only compute loss on unmasked part
    """
    masks = (y_true != null_val).float()
    masks /= masks.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * masks
    # trick for nans:
    # https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()


def masked_mse_loss(y_pred, y_true, null_val=0.0):
    """
    Only compute MSE loss on unmasked part
    """
    masks = (y_true != null_val).float()
    masks /= masks.mean()
    loss = (y_pred - y_true).pow(2)
    loss = loss * masks
    # trick for nans:
    # https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    loss = torch.sqrt(torch.mean(loss))
    return loss


def compute_regression_loss(
    y_true,
    y_predicted,
    standard_scaler=None,
    device=None,
    loss_fn="mae",
    mask_val=0.0,
    is_tensor=True,
):
    """
    Compute masked MAE loss with inverse scaled y_true and y_predict
    Args:
        y_true: ground truth signals, shape (batch_size, mask_len, num_nodes, feature_dim)
        y_predicted: predicted signals, shape (batch_size, mask_len, num_nodes, feature_dim)
        standard_scaler: class StandardScaler object
        device: device
        mask: int, masked node ID
        loss_fn: 'mae' or 'mse'
        is_tensor: whether y_true and y_predicted are PyTorch tensor
    """
    if device is not None:
        y_true = y_true.to(device)
        y_predicted = y_predicted.to(device)

    if standard_scaler is not None:
        y_true = standard_scaler.inverse_transform(
            y_true, is_tensor=is_tensor, device=device
        )

        y_predicted = standard_scaler.inverse_transform(
            y_predicted, is_tensor=is_tensor, device=device
        )

    if loss_fn == "mae":
        return masked_mae_loss(y_predicted, y_true, mask_val=mask_val)
    else:
        return masked_mse_loss(y_predicted, y_true, mask_val=mask_val)


""" Metrics for ICBEB ECG dataset from https://github.com/helme/ecg_ptbxl_benchmarking/blob/master/code/utils/utils.py """


def find_optimal_cutoff_threshold_for_beta(target, predicted, n_thresholds=100, threshold_on="gbeta"):
    thresholds = np.linspace(0.00, 1, n_thresholds)
    scores = [
        challenge_metrics(target, predicted > t, single=True)["G_beta_macro"]
        for t in thresholds
    ]
    optimal_idx = np.argmax(scores)
    return thresholds[optimal_idx]


def find_optimal_cutoff_thresholds_for_beta(y_true, y_pred, threshold_on="gbeta"):
    return [
        find_optimal_cutoff_threshold_for_beta(
            y_true[:, k][:, np.newaxis], y_pred[:, k][:, np.newaxis], threshold_on=threshold_on
        )
        for k in range(y_true.shape[1])
    ]

def thresh_max_f1(y_true, y_prob):
    """
    Find best threshold based on precision-recall curve to maximize F1-score.
    Binary calssification only
    """
    if len(np.unique(y_true)) > 2:
        raise NotImplementedError

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thresh_filt = []
    fscore = []
    n_thresh = len(thresholds)
    for idx in range(n_thresh):
        curr_f1 = (2 * precision[idx] * recall[idx]) / (precision[idx] + recall[idx])
        if not (np.isnan(curr_f1)):
            fscore.append(curr_f1)
            thresh_filt.append(thresholds[idx])
    # locate the index of the largest f score
    ix = np.argmax(np.array(fscore))
    best_thresh = thresh_filt[ix]
    return best_thresh

def find_optimal_cutoff_thresholds_for_f1(y_true, y_prob):
    return [
        thresh_max_f1(
            y_true[:, k][:, np.newaxis], y_prob[:, k][:, np.newaxis]
        )
        for k in range(y_true.shape[1])
    ]

def thresh_youden(y_true, y_prob):
    """Determine cutoff threshold based on Youden's J-Index"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youdenJ = tpr - fpr
    optimal_idx = np.argmax(youdenJ)
    optimal_thresh = thresholds[optimal_idx]
    return optimal_thresh

def find_optimal_cutoff_thresholds_youden(y_true, y_prob):
    return [
        thresh_youden(
            y_true[:, k][:, np.newaxis], y_prob[:, k][:, np.newaxis]
        )
        for k in range(y_true.shape[1])
    ]


def apply_thresholds(preds, thresholds):
    """
    apply class-wise thresholds to prediction score in order to get binary format.
    BUT: if no score is above threshold, pick maximum. This is needed due to metric issues.
    """
    tmp = []
    for p in preds:
        tmp_p = (p > thresholds).astype(int)
        if np.sum(tmp_p) == 0:
            tmp_p[np.argmax(p)] = 1
        tmp.append(tmp_p)
    tmp = np.array(tmp)
    return tmp


def challenge_metrics(
    y_true, y_pred, beta1=2, beta2=2, class_weights=None, single=False
):
    """Adapted from https://github.com/helme/ecg_ptbxl_benchmarking/blob/master/code/utils/utils.py"""
    f_beta = 0
    g_beta = 0
    f_beta_all = []
    g_beta_all = []
    for classi in range(y_true.shape[1]):
        y_truei, y_predi = y_true[:, classi], y_pred[:, classi]
        TP, FP, TN, FN = 0.0, 0.0, 0.0, 0.0
        for i in range(len(y_predi)):
            if y_truei[i] == y_predi[i] == 1:
                TP += 1.0
            if (y_predi[i] == 1) and (y_truei[i] != y_predi[i]):
                FP += 1.0
            if y_truei[i] == y_predi[i] == 0:
                TN += 1.0
            if (y_predi[i] == 0) and (y_truei[i] != y_predi[i]):
                FN += 1.0
        f_beta_i = ((1 + beta1**2) * TP) / (
            (1 + beta1**2) * TP + FP + (beta1**2) * FN
        )
        g_beta_i = (TP) / (TP + FP + beta2 * FN)

        f_beta += f_beta_i
        g_beta += g_beta_i

        f_beta_all.append(f_beta_i)
        g_beta_all.append(g_beta_i)

    return {
        "F_beta_macro": f_beta / y_true.shape[1],
        "G_beta_macro": g_beta / y_true.shape[1],
        "F_beta_all": np.array(f_beta_all),
        "G_beta_all": np.array(g_beta_all),
    }

def compute_fbeta_gbeta(y_true, y_pred):
    # PhysioNet/CinC Challenges metrics
    challenge_scores = challenge_metrics(y_true, y_pred, beta1=2, beta2=2)
    fbeta = challenge_scores["F_beta_all"]
    gbeta = challenge_scores["G_beta_all"]

    return fbeta, gbeta