import sys
import git
import os
import pytorch_lightning as pl
import pickle
import numpy as np
import h5py
import pandas as pd
import torch
import torch_geometric

from collections import Counter
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Data, Dataset
from typing import Optional
from tqdm import tqdm

from constants import DODH_CHANNELS
from data.data_utils.general_data_utils import StandardScaler, ImbalancedDatasetSampler
from data.data_utils.sleep_data_utils import read_dreem_data

DODH_FILEMARKER_DIR = "data/file_markers_dodh"


def reorder_channels(channels, dataset_name):

    if dataset_name == "dodh":
        channel_idxs = np.array([channels.index(ch) for ch in DODH_CHANNELS])
    else:
        raise NotImplementedError

    return channel_idxs


class DreemDataset(Dataset):
    def __init__(
        self,
        root,
        raw_data_path,
        file_marker,
        split,
        dataset_name,
        freq,
        scaler=None,
        transform=None,
        pre_transform=None,
    ):
        self.root = root
        self.raw_data_path = raw_data_path
        self.file_marker = file_marker
        self.split = split
        self.seq_len = 30  # hard-coded
        self.num_nodes = len(DODH_CHANNELS)
        self.scaler = scaler
        self.dataset_name = dataset_name
        self.freq = freq

        self.df_file = file_marker
        self.records = self.df_file["record_id"].tolist()
        self.labels = self.df_file["label"].tolist()
        self.clip_idxs = self.df_file["clip_index"].tolist()

        # process
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [
            os.path.join(self.raw_data_path, fn)
            for fn in os.listdir(self.raw_data_path)
        ]

    def len(self):
        return len(self.df_file)

    def get_labels(self):
        return torch.FloatTensor(self.labels)

    def get(self, idx):

        h5_file_name = self.records[idx]
        y = self.labels[idx]
        clip_idx = int(self.df_file.iloc[idx]["clip_index"])

        writeout_fn = h5_file_name.split(".h5")[0] + "_" + str(clip_idx)

        # read data
        try:
            signals, channels, _ = read_dreem_data(self.raw_data_path, h5_file_name)
        except:
            with h5py.File(os.path.join(self.raw_data_path, h5_file_name), "r") as hf:
                signals = hf["signals"][:]
                signals = np.transpose(
                    signals, (1, 0)
                )  # (total_seq_len*freq, num_channels)
                channels = hf["channels"][:]
                channels = [ch.decode("UTF-8") for ch in channels]
                fs = hf["fs"][()]
                assert self.freq == fs

        physical_len = int(self.freq * self.seq_len)
        start_idx = clip_idx * physical_len
        end_idx = start_idx + physical_len

        channel_idxs = reorder_channels(channels, self.dataset_name)

        x = signals[channel_idxs, start_idx:end_idx]        
        x = torch.FloatTensor(x).unsqueeze(-1)  # (num_channels, seq_len*freq, 1)
        y = torch.LongTensor([y])

        if self.scaler is not None:
            # standardize
            x = self.scaler.transform(x)

        # pyg graph
        data = Data(x=x.float(), y=y, writeout_fn=writeout_fn)

        return data


class Dreem_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        raw_data_path,
        dataset_name,
        freq,
        train_batch_size,
        test_batch_size,
        num_workers,
        standardize=True,
        balanced_sampling=False,
        use_class_weight=False,
        pin_memory=False,
    ):
        super().__init__()

        if use_class_weight and balanced_sampling:
            raise ValueError(
                "Choose only one of use_class_weight or balanced_sampling!"
            )

        self.raw_data_path = raw_data_path
        self.dataset_name = dataset_name
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.standardize = standardize
        self.balanced_sampling = balanced_sampling
        self.use_class_weight = use_class_weight
        self.pin_memory = pin_memory
        self.num_nodes = len(DODH_CHANNELS)

        self.file_markers = {}
        for split in ["train", "val", "test"]:
            if dataset_name == "dodh":
                print("{}_file_markers.csv".format(split))
                self.file_markers[split] = pd.read_csv(
                    os.path.join(
                        DODH_FILEMARKER_DIR, "{}_file_markers.csv".format(split)
                    )
                )   
            else:
                raise NotImplementedError()

        if standardize:
            train_files = list(set(self.file_markers["train"]["record_id"].tolist()))
            self.mean, self.std = self._compute_mean_std(
                train_files, num_nodes=self.num_nodes
            )
            print("mean:", self.mean.shape)

            self.scaler = StandardScaler(mean=self.mean, std=self.std)
        else:
            self.scaler = None

        self.train_dataset = DreemDataset(
            root=None,
            raw_data_path=self.raw_data_path,
            file_marker=self.file_markers["train"],
            split="train",
            dataset_name=dataset_name,
            freq=freq,
            scaler=self.scaler,
            transform=None,
            pre_transform=None,
        )

        # compute class weights
        if use_class_weight:
            self.class_weights = torch.FloatTensor(
                [
                    np.sum(self.train_dataset.labels == c) / len(self.train_dataset)
                    for c in np.arange(5)
                ]
            )
            self.class_weights /= torch.sum(self.class_weights)
            print("Class weight:", self.class_weights)
        else:
            self.class_weights = None

        self.val_dataset = DreemDataset(
            root=None,
            raw_data_path=self.raw_data_path,
            file_marker=self.file_markers["val"],
            split="val",
            dataset_name=dataset_name,
            freq=freq,
            scaler=self.scaler,
            transform=None,
            pre_transform=None,
        )

        self.test_dataset = DreemDataset(
            root=None,
            raw_data_path=self.raw_data_path,
            file_marker=self.file_markers["test"],
            split="test",
            dataset_name=dataset_name,
            freq=freq,
            scaler=self.scaler,
            transform=None,
            pre_transform=None,
        )

    def train_dataloader(self):

        if self.balanced_sampling:
            class_counts = list(
                Counter(self.train_dataset.get_labels().cpu().numpy()).values()
            )
            min_samples = np.min(np.array(class_counts))
            sampler = ImbalancedDatasetSampler(
                dataset=self.train_dataset,
                num_samples=min_samples * 5,
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

    def _compute_mean_std(self, train_files, num_nodes):
        count = 0
        signal_sum = np.zeros((num_nodes))
        signal_sum_sqrt = np.zeros((num_nodes))
        print("Computing mean and std of training data...")
        for idx in tqdm(range(len(train_files))):
            try:
                signal, channels, _ = read_dreem_data(
                    self.raw_data_path, train_files[idx]
                )
            except:
                with h5py.File(
                    os.path.join(self.raw_data_path, train_files[idx]), "r"
                ) as hf:
                    signal = hf["signals"][:]
                    signal = np.transpose(
                        signal, (1, 0)
                    )  # (total_seq_len*freq, num_channels)
                    channels = hf["channels"][:]
                    channels = [ch.decode("UTF-8") for ch in channels]
                    fs = hf["fs"][()]
            channel_idxs = reorder_channels(channels, self.dataset_name)
            signal = signal[channel_idxs, :]
            signal_sum += signal.sum(axis=-1)
            signal_sum_sqrt += (signal**2).sum(axis=-1)
            count += signal.shape[-1]
        total_mean = signal_sum / count
        total_var = (signal_sum_sqrt / count) - (total_mean**2)
        total_std = np.sqrt(total_var)

        return np.expand_dims(np.expand_dims(total_mean, -1), -1), np.expand_dims(
            np.expand_dims(total_std, -1), -1
        )

    def teardown(self, stage=None):
        # clean up after fit or test
        # called on every process in DDP
        pass
