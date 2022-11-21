import sys
import os
import pytorch_lightning as pl
import pickle
import numpy as np
import h5py
import pandas as pd
import torch
import torch_geometric

from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Data, Dataset
from typing import Optional
from tqdm import tqdm
from constants import TUH_FREQUENCY as FREQ
from data.data_utils.general_data_utils import StandardScaler, ImbalancedDatasetSampler

FILEMARKER_DIR = "data/file_markers_tuh_v1.5.2"


class TUHDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        raw_data_path,
        file_marker,
        split,
        seq_len,
        num_nodes,
        adj_mat_dir,
        scaler=None,
        transform=None,
        pre_transform=None,
        repreproc=False,
    ):
        self.root = root
        self.raw_data_path = raw_data_path
        self.file_marker = file_marker
        self.split = split
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.adj_mat_dir = adj_mat_dir
        self.scaler = scaler

        self.df_file = file_marker
        self.file_names = self.df_file["file_name"].tolist()
        self.labels = self.df_file["is_seizure"].tolist()
        self.clip_idxs = self.df_file["clip_index"].tolist()

        # process
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [
            os.path.join(self.raw_data_path, fn)
            for fn in os.listdir(self.raw_data_path)
        ]

    @property
    def processed_file_names(self):
        return ["{}_{}.pt".format(self.file_names[idx].split(".h5")[0], self.clip_idxs[idx]) for idx in range(len(self.df_file))]

    def len(self):
        return len(self.file_names)

    def _get_combined_graph(self):
        with open(self.adj_mat_dir, "rb") as pf:
            adj_mat = pickle.load(pf)
            adj_mat = adj_mat[-1]
        return adj_mat

    def get_labels(self):
        return torch.FloatTensor(self.labels)

    def process(self):
        for idx in tqdm(range(len(self.file_names))):

            h5_file_name = self.file_names[idx]
            y = self.labels[idx]
            clip_idx = int(self.df_file.iloc[idx]["clip_index"])

            writeout_fn = h5_file_name.split(".h5")[0] + "_" + str(clip_idx)

            if os.path.exists(
                os.path.join(self.processed_dir, "{}.pt".format(writeout_fn))
            ):
                continue

            with h5py.File(os.path.join(self.raw_data_path, h5_file_name), "r") as hf:
                x = hf["resampled_signal"][()]  # (num_nodes, time * freq)
            time_start_idx = clip_idx * int(FREQ * self.seq_len)
            time_end_idx = time_start_idx + int(FREQ * self.seq_len)

            x = x[:, time_start_idx:time_end_idx]  # (num_nodes, seq_len*freq)

            assert x.shape[1] == FREQ * self.seq_len
            x = np.expand_dims(x, axis=-1)  # (num_nodes, seq_len*freq, 1)

            # get edge index
            adj_mat = self._get_combined_graph()
            edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(
                torch.FloatTensor(adj_mat)
            )

            # pyg graph
            x = torch.FloatTensor(x)  # (num_nodes, seq_len*freq, 1)
            y = torch.FloatTensor([y])
            data = Data(
                x=x,
                edge_index=edge_index.contiguous(),
                edge_attr=edge_weight,
                y=y,
                adj_mat=torch.FloatTensor(adj_mat).unsqueeze(0),
            )

            data.writeout_fn = writeout_fn

            torch.save(
                data,
                os.path.join(self.processed_dir, "{}.pt".format(writeout_fn)),
            )

    def get(self, idx):

        h5_file_name = self.file_names[idx]
        y = self.labels[idx]
        clip_idx = int(self.df_file.iloc[idx]["clip_index"])

        writeout_fn = h5_file_name.split(".h5")[0] + "_" + str(clip_idx)

        data = torch.load(os.path.join(self.processed_dir, "{}.pt".format(writeout_fn)))

        if self.scaler is not None:
            # standardize
            data.x = self.scaler.transform(data.x)

        data.x = data.x.float()

        return data


class TUH_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        raw_data_path,
        preproc_save_dir,
        seq_len,
        num_nodes,
        train_batch_size,
        test_batch_size,
        num_workers,
        adj_mat_dir=None,
        standardize=True,
        balanced_sampling=False,
        pin_memory=False,
    ):
        super().__init__()

        self.raw_data_path = raw_data_path
        self.preproc_save_dir = preproc_save_dir
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.adj_mat_dir = adj_mat_dir
        self.standardize = standardize
        self.balanced_sampling = balanced_sampling
        self.pin_memory = pin_memory

        self.file_markers = {}
        for split in ["train", "val", "test"]:
            self.file_markers[split] = pd.read_csv(
                os.path.join(
                    FILEMARKER_DIR, "{}_file_markers_{}s.csv".format(split, seq_len),
                ),
            )

        if standardize:
            train_files = list(set(self.file_markers["train"]["file_name"].tolist()))
            train_files = [os.path.join(raw_data_path, fn) for fn in train_files]
            self.mean, self.std = self._compute_mean_std(
                train_files, num_nodes=num_nodes
            )
            print("mean:", self.mean.shape)

            self.scaler = StandardScaler(mean=self.mean, std=self.std)
        else:
            self.scaler = None

        self.train_dataset = TUHDataset(
            root=self.preproc_save_dir,
            raw_data_path=self.raw_data_path,
            file_marker=self.file_markers["train"],
            split="train",
            seq_len=self.seq_len,
            num_nodes=self.num_nodes,
            adj_mat_dir=self.adj_mat_dir,
            scaler=self.scaler,
            transform=None,
            pre_transform=None,
        )

        self.val_dataset = TUHDataset(
            root=self.preproc_save_dir,
            raw_data_path=self.raw_data_path,
            file_marker=self.file_markers["val"],
            split="val",
            seq_len=self.seq_len,
            num_nodes=self.num_nodes,
            adj_mat_dir=self.adj_mat_dir,
            scaler=self.scaler,
            transform=None,
            pre_transform=None,
        )

        self.test_dataset = TUHDataset(
            root=self.preproc_save_dir,
            raw_data_path=self.raw_data_path,
            file_marker=self.file_markers["test"],
            split="test",
            seq_len=self.seq_len,
            num_nodes=self.num_nodes,
            adj_mat_dir=self.adj_mat_dir,
            scaler=self.scaler,
            transform=None,
            pre_transform=None,
        )

    def train_dataloader(self):

        if self.balanced_sampling:
            num_pos = torch.sum(self.train_dataset.get_labels() == 1)
            sampler = ImbalancedDatasetSampler(
                dataset=self.train_dataset,
                num_samples=num_pos * 2,
                replacement=False,
            )
            shuffle = False

        else:
            sampler = None
            shuffle = True

        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            sampler=sampler,
            shuffle=shuffle,
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

    def _compute_mean_std(self, train_files, num_nodes=19):
        if ".h5" in train_files[0]:
            count = 0
            signal_sum = np.zeros((num_nodes))
            signal_sum_sqrt = np.zeros((num_nodes))
            print("Computing mean and std of training data...")
            for idx in tqdm(range(len(train_files))):
                with h5py.File(train_files[idx], "r") as hf:
                    signal = hf["resampled_signal"][()]  # (num_nodes, time * freq)
                signal_sum += signal.sum(axis=-1)
                signal_sum_sqrt += (signal**2).sum(axis=-1)
                count += signal.shape[-1]
            total_mean = signal_sum / count
            total_var = (signal_sum_sqrt / count) - (total_mean**2)
            total_std = np.sqrt(total_var)
        else:
            raise NotImplementedError

        return np.expand_dims(np.expand_dims(total_mean, -1), -1), np.expand_dims(
            np.expand_dims(total_std, -1), -1
        )

    def teardown(self, stage=None):
        # clean up after fit or test
        # called on every process in DDP
        pass