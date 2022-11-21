import numpy as np
import os
import pickle
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd

from data.datamodules.datamodule_tuh import TUH_DataModule
from data.datamodules.datamodule_traffic import Traffic_DataModule
from data.datamodules.datamodule_dreem import Dreem_DataModule

from args import get_args
import torch
from model.graphs4mer import *
from model.temporal_gnn import *
from model.lstm import LSTMModel
from model.s4 import S4Model
from constants import *

import utils.utils as utils
from utils.schedulers import *
from tqdm import tqdm
from dotted_dict import DottedDict
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
import copy
from collections import OrderedDict
from json import dumps
import sys
import itertools


class PLModel(pl.LightningModule):
    def __init__(
        self,
        args,
        lr=1e-3,
        weight_decay=1e-3,
        optimizer_name="adamw",
        scheduler_name="cosine",
        steps_per_epoch=None,
        scaler=None,
        log_prefix="",
        **scheduler_kwargs,
    ):
        super().__init__()
        self.args = args
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.steps_per_epoch = steps_per_epoch
        self.scaler = scaler
        self.scheduler_kwargs = scheduler_kwargs
        self.log_prefix = log_prefix

        self._build_model()

    def _build_model(self):
        args = self.args
        undirected_graph = (
            False if (args.dataset == "pems_bay") else True
        )
        if args.dataset == "tuh":
            args.max_seq_len *= TUH_FREQUENCY
        elif args.dataset == "dodh":
            args.max_seq_len *= args.sampling_freq
        if args.model_name.lower() == "graphs4mer":
            if args.task == "classification":
                self.model = GraphS4mer(
                    input_dim=args.input_dim,
                    num_nodes=args.num_nodes,
                    dropout=args.dropout,
                    num_temporal_layers=args.num_temporal_layers,
                    g_conv=args.g_conv,
                    num_gnn_layers=args.num_gcn_layers,
                    hidden_dim=args.hidden_dim,
                    max_seq_len=args.max_seq_len,
                    resolution=args.resolution,
                    state_dim=args.state_dim,
                    channels=args.channels,
                    temporal_model=args.temporal_model,
                    bidirectional=args.bidirectional,
                    temporal_pool=args.temporal_pool,
                    prenorm=args.prenorm,
                    postact=args.postact,
                    metric=args.graph_learn_metric,
                    adj_embed_dim=args.adj_embed_dim,
                    gin_mlp=args.gin_mlp,
                    train_eps=args.train_eps,
                    prune_method=args.prune_method,
                    edge_top_perc=args.edge_top_perc,
                    thresh=args.thresh,
                    graph_pool=args.graph_pool,
                    activation_fn=args.activation_fn,
                    num_classes=args.output_dim,
                    undirected_graph=undirected_graph,
                    use_prior=args.use_prior,
                    K=args.knn,
                    regularizations=args.regularizations,
                    residual_weight=args.residual_weight,
                )
            else:
                self.model = GraphS4mer_Regression(
                    input_dim=args.input_dim,
                    output_dim=args.output_dim,
                    num_nodes=args.num_nodes,
                    dropout=args.dropout,
                    num_temporal_layers=args.num_temporal_layers,
                    g_conv=args.g_conv,
                    num_gnn_layers=args.num_gcn_layers,
                    hidden_dim=args.hidden_dim,
                    max_seq_len=args.max_seq_len,
                    output_seq_len=args.output_seq_len,
                    resolution=args.resolution,
                    state_dim=args.state_dim,
                    channels=args.channels,
                    temporal_model=args.temporal_model,
                    bidirectional=args.bidirectional,
                    prenorm=args.prenorm,
                    postact=args.postact,
                    metric=args.graph_learn_metric,
                    adj_embed_dim=args.adj_embed_dim,
                    gin_mlp=args.gin_mlp,
                    train_eps=args.train_eps,
                    graph_pool=args.graph_pool,
                    prune_method=args.prune_method,
                    edge_top_perc=args.edge_top_perc,
                    thresh=args.thresh,
                    activation_fn=args.activation_fn,
                    undirected_graph=undirected_graph,
                    use_prior=args.use_prior,
                    K=args.knn,
                    regularizations=args.regularizations,
                    residual_weight=args.residual_weight,
                )
        elif args.model_name.lower() == "temporal_gnn":
            if args.task == "classification":
                self.model = TemporalGNN(
                    input_dim=args.input_dim,
                    num_nodes=args.num_nodes,
                    dropout=args.dropout,
                    g_conv=args.g_conv,
                    num_gnn_layers=args.num_gcn_layers,
                    hidden_dim=args.hidden_dim,
                    max_seq_len=args.max_seq_len,
                    num_temporal_layers=args.num_temporal_layers,
                    state_dim=args.state_dim,
                    channels=args.channels,
                    temporal_model=args.temporal_model,
                    bidirectional=args.bidirectional,
                    temporal_pool=args.temporal_pool,
                    prenorm=args.prenorm,
                    postact=args.postact,
                    gin_mlp=args.gin_mlp,
                    train_eps=args.train_eps,
                    graph_pool=args.graph_pool,
                    activation_fn=args.activation_fn,
                    num_classes=args.output_dim,
                    undirected_graph=undirected_graph,
                    use_prior=args.use_prior,
                    K=args.knn,
                )
            else:
                self.model = TemporalGNN_Regression(
                    input_dim=args.input_dim,
                    output_dim=args.output_dim,
                    num_nodes=args.num_nodes,
                    dropout=args.dropout,
                    g_conv=args.g_conv,
                    num_gnn_layers=args.num_gcn_layers,
                    hidden_dim=args.hidden_dim,
                    max_seq_len=args.max_seq_len,
                    output_seq_len=args.output_seq_len,
                    num_temporal_layers=args.num_temporal_layers,
                    state_dim=args.state_dim,
                    channels=args.channels,
                    temporal_model=args.temporal_model,
                    bidirectional=args.bidirectional,
                    prenorm=args.prenorm,
                    postact=args.postact,
                    gin_mlp=args.gin_mlp,
                    train_eps=args.train_eps,
                    activation_fn=args.activation_fn,
                    undirected_graph=undirected_graph,
                    use_prior=args.use_prior,
                    K=args.knn,
                )
        elif args.model_name.lower() == "s4":
            self.model = S4Model(
                d_input=args.num_nodes * args.input_dim,
                d_output=args.output_dim if (args.task != "regression") else (args.output_dim * args.num_nodes),
                d_model=args.hidden_dim,
                d_state=args.state_dim,
                n_layers=args.num_temporal_layers,
                dropout=args.dropout,
                prenorm=args.prenorm,
                l_max=args.max_seq_len,
                l_output=args.output_seq_len,
                bidirectional=args.bidirectional,
                postact=args.postact,  # none or 'glu'
                add_decoder=True,
                pool=False,  # hard-coded
                temporal_pool=args.temporal_pool,
            )
        elif args.model_name.lower() == "lstm":
            self.model = LSTMModel(
                input_dim=args.num_nodes * args.input_dim,
                hidden_dim=args.hidden_dim,
                num_rnn_layers=args.num_temporal_layers,
                output_dim=args.output_dim if (args.task != "regression") else (args.output_dim * args.num_nodes),
                output_seq_len=args.output_seq_len,
                temporal_pool=args.temporal_pool,
                dropout=args.dropout,
                add_decoder=True,
            )
        else:
            raise NotImplementedError

    def _aggregate_regularization_losses(self, reg_loss_dict):
        reg_loss = 0.0
        for k in self.args.regularizations:
            if k == "feature_smoothing":
                reg_loss = (
                    reg_loss + self.args.feature_smoothing_weight * reg_loss_dict[k]
                )
            elif k == "degree":
                reg_loss = reg_loss + self.args.degree_weight * reg_loss_dict[k]
            elif k == "sparse":
                reg_loss = reg_loss + self.args.sparse_weight * reg_loss_dict[k]
            else:
                raise NotImplementedError()
        return reg_loss

    def training_step(self, batch, batch_idx):
        logits, y, cls_loss, reg_loss, _, _, _ = self._shared_step(batch)

        log_dict = {}
        if "graphs4mer" in self.args.model_name:
            loss = cls_loss + reg_loss
            log_dict["{}train/reg_loss".format(self.log_prefix)] = reg_loss.item()
        else:
            loss = cls_loss

        log_dict["{}train/cls_loss".format(self.log_prefix)] = cls_loss.item()
        log_dict["{}train/loss".format(self.log_prefix)] = loss.item()

        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):

        logits, y, cls_loss, reg_loss, file_names, _, _ = self._shared_step(batch)

        return {
            "labels": y,
            "logits": logits,
            "reg_loss": reg_loss,
            "file_names": file_names,
        }

    def validation_epoch_end(self, outputs):
        logits = torch.cat([output["logits"] for output in outputs]).squeeze()
        labels = torch.cat([output["labels"] for output in outputs]).squeeze()

        log_dict = {}

        if "graphs4mer" in self.args.model_name:
            reg_loss = torch.mean(
                torch.stack([output["reg_loss"] for output in outputs])
            )
        if self.args.task == "classification":
            # classification
            if self.args.output_dim == 1:
                cls_loss = F.binary_cross_entropy_with_logits(
                    logits,
                    labels,
                    pos_weight=torch.FloatTensor([self.args.pos_weight]).to(
                        self.device
                    ),
                )
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)
            else:
                cls_loss = F.cross_entropy(logits, labels.long())
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=-1)
            scores_dict = utils.eval_dict(
                y_pred=preds,
                y=labels.cpu().numpy(),
                y_prob=probs,
                average=self.args.metric_avg,
                metrics=self.args.eval_metrics,
            )
        else:
            cls_loss = F.l1_loss(logits, labels, reduction="mean")  # MAE loss
            scores_dict = utils.eval_dict(
                y_pred=logits.cpu().numpy(),
                y=labels.cpu().numpy(),
                metrics=self.args.eval_metrics,
            )

        if "graphs4mer" in self.args.model_name:
            loss = cls_loss + reg_loss
            log_dict["{}val/reg_loss".format(self.log_prefix)] = reg_loss.item()
        else:
            loss = cls_loss

        log_dict["{}val/cls_loss".format(self.log_prefix)] = cls_loss.item()
        log_dict["{}val/loss".format(self.log_prefix)] = loss.item()
        for k, v in scores_dict.items():
            log_dict["{}val/{}".format(self.log_prefix, k)] = v

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx, dataloader_idx=0):

        # assumes validation loader first, then test loader
        prefix = ["val", "test"][dataloader_idx]

        (
            logits,
            y,
            cls_loss,
            reg_loss,
            file_names,
            raw_attn_weight,
            adj_mat_learned,
        ) = self._shared_step(batch)
        return {
            "labels": y,
            "logits": logits,
            "prefix": prefix,
            "file_names": file_names,
            "raw_attn_weight": raw_attn_weight,
            "adj_mat_learned": adj_mat_learned,
        }

    def test_epoch_end(self, outputs):

        for curr_outputs in outputs:
            logits = torch.cat([output["logits"] for output in curr_outputs]).squeeze()
            labels = torch.cat([output["labels"] for output in curr_outputs]).squeeze()
            file_names = [output["file_names"] for output in curr_outputs]
            prefix = [output["prefix"] for output in curr_outputs][0]

            if self.args.task == "classification":
                # classification
                if self.args.output_dim == 1:
                    cls_loss = F.binary_cross_entropy_with_logits(
                        logits,
                        labels,
                        pos_weight=torch.FloatTensor([self.args.pos_weight]).to(
                            self.device
                        ),
                    )
                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds = (probs > 0.5).astype(int)
                else:
                    cls_loss = F.cross_entropy(logits, labels.long())
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    preds = np.argmax(probs, axis=-1)

                scores_dict = utils.eval_dict(
                    y_pred=preds,
                    y=labels.cpu().numpy(),
                    y_prob=probs,
                    average=self.args.metric_avg,
                    metrics=self.args.eval_metrics,
                )
            else:
                # regression tasks
                scores_dict = utils.eval_dict(
                    y_pred=logits.cpu().numpy(),
                    y=labels.cpu().numpy(),
                    metrics=self.args.eval_metrics,
                )

            # log
            res_str = "{} - ".format(prefix)
            for k, v in scores_dict.items():
                res_str = res_str + "{}: {:.4f}; ".format(k, v)
            print(res_str)

            self.log_dict(
                {
                    "{}{}/best_{}".format(
                        self.log_prefix, prefix, self.args.metric_name
                    ): scores_dict[self.args.metric_name]
                },
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
                sync_dist=True,
            )

            # save scores
            with open(
                os.path.join(self.args.save_dir, "{}_scores.pkl".format(prefix)), "wb"
            ) as pf:
                pickle.dump(scores_dict, pf)

            # save outputs
            if self.args.save_output:
                outputs_dict = {
                    "logits": logits,
                    "labels": labels,
                    "file_names": file_names,
                }
                if "self_attention" in self.args.model_name:
                    raw_attn_weight = torch.cat(
                        [output["raw_attn_weight"] for output in curr_outputs]
                    )
                    adj_mat_learned = torch.cat(
                        [output["adj_mat_learned"] for output in curr_outputs]
                    )
                    outputs_dict["raw_attn_weight"] = raw_attn_weight
                    outputs_dict["adj_mat_learned"] = adj_mat_learned

                with open(
                    os.path.join(self.args.save_dir, "{}_results.pkl".format(prefix)),
                    "wb",
                ) as pf:
                    pickle.dump(outputs_dict, pf)

    def _shared_step(self, batch):
        y = batch.y
        if y.shape[-1] == 1:
            y = y.view(-1)

        raw_attn_weight = []
        adj_mat_learned = []
        reg_loss = None
        if "graphs4mer" in self.args.model_name:
            if self.args.save_output:
                logits, reg_loss_dict, raw_attn_weight, adj_mat_learned = self.model(
                    batch, return_attention=True
                )
            else:
                logits, reg_loss_dict = self.model(batch, return_attention=False)

            reg_loss = self._aggregate_regularization_losses(reg_loss_dict)
        elif self.args.model_name == "s4" or self.args.model_name == "lstm":
            x = (
                batch.x.reshape(
                    -1,
                    self.args.num_nodes,
                    self.args.max_seq_len,
                    self.args.input_dim,
                )
                .transpose(1, 2)
                .reshape(
                    -1, self.args.max_seq_len, self.args.num_nodes * self.args.input_dim
                )
            )  # (batch, seq_len, num_nodes*input_dim)
            logits = self.model(x)
        else:
            logits = self.model(batch)

        if self.args.task == "classification":
            # classification task
            if self.args.output_dim == 1:
                cls_loss = F.binary_cross_entropy_with_logits(
                    logits.view(-1),
                    y,
                    pos_weight=torch.FloatTensor([self.args.pos_weight]).to(
                        self.device
                    ),
                )
            else:
                cls_loss = F.cross_entropy(logits, y.long())
        else:
            # regression task
            y = y.reshape(
                -1, self.args.num_nodes, self.args.max_seq_len, self.args.input_dim
            )  # (batch, num_nodes, seq_len, input_dim)
            y = y[
                ..., : self.args.output_dim
            ]  # (batch, num_nodes, seq_len, output_dim)

            if self.args.model_name == "s4" or self.args.model_name == "lstm":
                logits = logits.reshape(
                    -1,
                    self.args.max_seq_len,
                    self.args.num_nodes,
                    self.args.output_dim,
                ).transpose(
                    1, 2
                )  # (batch, num_nodes, seq_len, output_dim)
            else:
                logits = logits.reshape(
                    -1,
                    self.args.num_nodes,
                    self.args.max_seq_len,
                    self.args.output_dim,
                )  # (batch, num_nodes, seq_len, output_dim)

            logits = self.scaler.inverse_transform(logits)
            y = self.scaler.inverse_transform(y)
            cls_loss = F.l1_loss(logits, y, reduction="mean")  # MAE loss

        return (
            logits,
            y,
            cls_loss,
            reg_loss,
            batch.writeout_fn,
            raw_attn_weight,
            adj_mat_learned,
        )

    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            optimizer = optim.Adam(
                params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "adamw":
            optimizer = optim.AdamW(
                params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            raise NotImplementedError

        if self.scheduler_name == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.args.num_epochs)
        elif self.scheduler_name == "one_cycle":
            print("steps_per_epoch:", self.steps_per_epoch)
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.lr,
                steps_per_epoch=self.steps_per_epoch,
                epochs=self.args.num_epochs,
            )
        elif self.scheduler_name == "timm_cosine":
            scheduler = TimmCosineLRScheduler(optimizer, **self.scheduler_kwargs)
        else:
            raise NotImplementedError

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main(args):

    # random seed
    pl.seed_everything(args.rand_seed, workers=True)

    # Get save directories
    args.save_dir = utils.get_save_dir(
        args.save_dir, training=True if args.do_train else False
    )
    save_dir = args.save_dir
    # Save args
    args_file = os.path.join(args.save_dir, "args.json")
    with open(args_file, "w") as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    # Build dataset
    print("Building dataset...")
    scaler = None
    if args.dataset == "tuh":
        datamodule = TUH_DataModule(
            preproc_save_dir=args.preproc_dir,
            raw_data_path=args.raw_data_dir,
            seq_len=args.max_seq_len,
            num_nodes=args.num_nodes,
            train_batch_size=args.train_batch_size,
            test_batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            adj_mat_dir=args.adj_mat_dir,
            standardize=False, # TODO: revert this!!
            balanced_sampling=args.balanced_sampling,
            pin_memory=True,
        )
    elif args.dataset == "pems_bay":
        datamodule = Traffic_DataModule(
            raw_data_dir=args.raw_data_dir,
            adj_mat_dir=args.adj_mat_dir,
            train_batch_size=args.train_batch_size,
            test_batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            standardize=True,
            pin_memory=True,
        )
        scaler = datamodule.scaler
    elif args.dataset == "dodh":
        datamodule = Dreem_DataModule(
            raw_data_path=args.raw_data_dir,
            dataset_name=args.dataset,
            freq=args.sampling_freq,
            train_batch_size=args.train_batch_size,
            test_batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            standardize=True,
            balanced_sampling=args.balanced_sampling,
            pin_memory=True,
        )
    else:
        raise NotImplementedError

    if args.load_model_path is not None:
        pl_model = PLModel.load_from_checkpoint(
            args.load_model_path,
            args=args,
            lr=args.lr_init,
            weight_decay=args.l2_wd,
            optimizer_name=args.optimizer,
            scheduler_name=args.scheduler,
            steps_per_epoch=len(datamodule.train_dataloader()),
            scaler=scaler,
            t_initial=args.t_initial,
            lr_min=args.lr_min,
            cycle_decay=args.cycle_decay,
            warmup_lr_init=args.warmup_lr_init,
            warmup_t=args.warmup_t,
            cycle_limit=args.cycle_limit,
        )
    else:
        pl_model = PLModel(
            args,
            lr=args.lr_init,
            weight_decay=args.l2_wd,
            optimizer_name=args.optimizer,
            scheduler_name=args.scheduler,
            steps_per_epoch=len(datamodule.train_dataloader()),
            scaler=scaler,
            t_initial=args.t_initial,
            lr_min=args.lr_min,
            cycle_decay=args.cycle_decay,
            warmup_lr_init=args.warmup_lr_init,
            warmup_t=args.warmup_t,
            cycle_limit=args.cycle_limit,
        )

    if args.do_train:
        checkpoint_callback = ModelCheckpoint(
            monitor="val/{}".format(args.metric_name),
            mode="max" if args.maximize_metric else "min",
            dirpath=args.save_dir,
            save_last=True,
            save_top_k=1,
            auto_insert_metric_name=False,
        )

        early_stopping_callback = EarlyStopping(
            monitor="val/loss", mode="min", patience=args.patience
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")

        if not (args.gpus > 1):
            trainer = pl.Trainer(
                accelerator="gpu",
                max_epochs=args.num_epochs,
                max_steps=-1,
                enable_progress_bar=True,
                callbacks=[
                    checkpoint_callback,
                    early_stopping_callback,
                    lr_monitor,
                ],
                benchmark=False,
                num_sanity_val_steps=0,
                devices=args.gpu_id,  # default to 1 GPU
                accumulate_grad_batches=args.accumulate_grad_batches,
            )
        else:
            # distributed data parallel
            trainer = pl.Trainer(
                accelerator="gpu",
                strategy=pl.strategies.DDPSpawnStrategy(
                    find_unused_parameters=False
                ),
                replace_sampler_ddp=False,
                max_epochs=args.num_epochs,
                max_steps=-1,
                enable_progress_bar=True,
                callbacks=[
                    checkpoint_callback,
                    early_stopping_callback,
                    lr_monitor,
                ],
                benchmark=False,
                num_sanity_val_steps=0,
                devices=torch.cuda.device_count(),
                accumulate_grad_batches=args.accumulate_grad_batches,
            )

        trainer.fit(pl_model, datamodule=datamodule)
        print("Training DONE.")

        # best val metric
        trainer.test(
            model=pl_model,
            ckpt_path="best",
            dataloaders=[datamodule.val_dataloader(), datamodule.test_dataloader()],
        )

    else:
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=args.gpu_id,
        )

        trainer.test(
            model=pl_model,
            ckpt_path=args.load_model_path,
            dataloaders=[datamodule.val_dataloader(), datamodule.test_dataloader()],
        )

if __name__ == "__main__":
    main(get_args())
