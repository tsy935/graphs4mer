import numpy as np
import torch
import pickle
import sys
import numpy as np
import os
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
from torch_geometric.loader import DataLoader
import torch_geometric
import pytorch_lightning as pl
import pandas as pd


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class TrafficDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        raw_data_dir,
        adj_mat_dir,
        split,
        standardize=True,
        transform=None,
        pre_transform=None,
        scaler=None,
    ):
        self.raw_data_dir = raw_data_dir
        self.adj_mat_dir = adj_mat_dir
        self.split = split
        self.standardize = standardize

        # node features
        data = np.load(raw_data_dir)
        self.x = data["x"]
        self.y = data["y"]

        # adj mat
        self.sensor_ids, self.sensor_id_to_ind, self.adj_mat = self._load_pickle(
            adj_mat_dir
        )
        self.num_nodes = len(self.sensor_ids)
        print("Number of nodes:", self.num_nodes)

        # scaler
        # following https://github.com/liyaguang/DCRNN/blob/master/lib/utils.py#L178-L194
        # only standardize feature dim 0 (traffic speed)
        if split == "train" and standardize:
            mean = self.x[..., 0].mean()
            std = self.x[..., 0].std()
            self.scaler = StandardScaler(mean=mean, std=std)
        else:
            self.scaler = scaler

        # process
        super().__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.raw_data_dir]

    @property
    def processed_file_names(self):
        return ["{}.dataset".format(self.split)]

    def len(self):
        return self.x.shape[0]

    def _load_pickle(self, pickle_file):
        try:
            with open(pickle_file, "rb") as f:
                pickle_data = pickle.load(f)
        except UnicodeDecodeError as e:
            with open(pickle_file, "rb") as f:
                pickle_data = pickle.load(f, encoding="latin1")
        except Exception as e:
            print("Unable to load data ", pickle_file, ":", e)
            raise
        return pickle_data

    def _get_labels(self):
        return self.y

    def process(self):
        data_list = []

        for idx in tqdm(range(self.x.shape[0])):
            x = self.x[idx]
            y = self.y[idx]

            if self.standardize:
                x[..., 0] = self.scaler.transform(x[..., 0])
                y[..., 0] = self.scaler.transform(y[..., 0])

            adj_mat = torch.FloatTensor(self.adj_mat)
            edge_index, edge_attr = torch_geometric.utils.dense_to_sparse(adj_mat)

            x = torch.FloatTensor(
                x.transpose(1, 0, 2)
            )  # (num_nodes, seq_len, input_dim)
            y = torch.FloatTensor(
                y.transpose(1, 0, 2)
            )  # (num_nodes, seq_len, input_dim)

            data = Data(
                x=x,
                y=y,
                edge_index=edge_index,
                edge_attr=edge_attr,
                adj_mat=adj_mat.unsqueeze(0),
                writeout_fn="", # dummy for consistency
            )
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class Traffic_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        raw_data_dir,
        adj_mat_dir,
        train_batch_size,
        test_batch_size,
        num_workers=4,
        standardize=True,
        transform=None,
        pre_transform=None,
        pin_memory=False,
        ddp=False,
    ):
        super().__init__()

        self.raw_data_dir = raw_data_dir
        self.adj_mat_dir = adj_mat_dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.standardize = standardize
        self.pin_memory = pin_memory
        self.ddp = ddp

        self.scaler = None
        self.train_dataset = TrafficDataset(
            root=raw_data_dir,
            raw_data_dir=os.path.join(raw_data_dir, "train.npz"),
            adj_mat_dir=adj_mat_dir,
            split="train",
            standardize=True,
            transform=None,
            pre_transform=None,
            scaler=None,
        )
        self.scaler = self.train_dataset.scaler

        self.val_dataset = TrafficDataset(
            root=raw_data_dir,
            raw_data_dir=os.path.join(raw_data_dir, "val.npz"),
            adj_mat_dir=adj_mat_dir,
            split="val",
            standardize=True,
            transform=None,
            pre_transform=None,
            scaler=self.scaler,
        )

        self.test_dataset = TrafficDataset(
            root=raw_data_dir,
            raw_data_dir=os.path.join(raw_data_dir, "test.npz"),
            adj_mat_dir=adj_mat_dir,
            split="test",
            standardize=True,
            transform=None,
            pre_transform=None,
            scaler=self.scaler,
        )

    def train_dataloader(self):

        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )
        return train_dataloader

    def val_dataloader(self):

        val_dataloader = DataLoader(
            dataset=self.val_dataset,
            shuffle=False,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )
        return val_dataloader

    def test_dataloader(self):

        test_dataloader = DataLoader(
            dataset=self.test_dataset,
            shuffle=False,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )
        return test_dataloader

    def teardown(self, stage=None):
        # clean up after fit or test
        # called on every process in DDP
        pass