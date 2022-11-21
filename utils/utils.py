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

def eval_dict(y_pred, y, y_prob=None, average="binary", metrics=[]):
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
        if len(np.unique(y)) <= 2:  # binary case
            scores_dict["auroc"] = roc_auc_score(y_true=y, y_score=y_prob)
        else:  # multiclass/multilabel case
            scores_dict["auroc"] = roc_auc_score(
                y_true=y, y_score=y_prob, average=average, multi_class="ovr"
            )
    if "kappa" in metrics:
        scores_dict["kappa"] = cohen_kappa_score(y1=y, y2=y_pred)
    if "mae" in metrics:
        scores_dict["mae"] = masked_mae_np(preds=y_pred, labels=y)
    if "rmse" in metrics:
        scores_dict["rmse"] = masked_rmse_np(preds=y_pred, labels=y)
    if "mse" in metrics:
        scores_dict["mse"] = masked_mse_np(preds=y_pred, labels=y)
    if "mape" in metrics:
        scores_dict["mape"] = masked_mape_np(preds=y_pred, labels=y)
    return scores_dict


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

def masked_mae_loss(y_pred, y_true, mask_val=0.0):
    """
    Only compute loss on unmasked part
    """
    masks = (y_true != mask_val).float()
    masks /= masks.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * masks
    # trick for nans:
    # https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()


def masked_mse_loss(y_pred, y_true, mask_val=0.0):
    """
    Only compute MSE loss on unmasked part
    """
    masks = (y_true != mask_val).float()
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