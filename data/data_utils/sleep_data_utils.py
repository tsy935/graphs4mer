import os
import numpy as np
import pandas as pd
import h5py


def read_dreem_data(data_dir, file_name):
    with h5py.File(os.path.join(data_dir, file_name), "r") as f:
        labels = f["hypnogram"][()]

        signals = []
        channels = []
        for key in f["signals"].keys():
            for ch in f["signals"][key].keys():
                signals.append(f["signals"][key][ch][()])
                channels.append(ch)
    signals = np.stack(signals, axis=0)

    return signals, channels, labels
