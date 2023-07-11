# Modeling Multivariate Biosignals With Graph Neural Networks and Structured State Space Models

Siyi Tang, Jared A Dunnmon, Qu Liangqiong, Khaled K Saab, Tina Baykaner, Christopher Lee-Messer, Daniel L Rubin. *Proceedings of the Conference on Health, Inference, and Learning*, PMLR 209:50-71, 2023. (**Best Paper Award**)

https://proceedings.mlr.press/v209/tang23a/tang23a.pdf

---
## Setup
This codebase requries python ≥ 3.9, pytorch ≥ 1.12.0, and pyg installed. Please refer to [PyTorch installation](https://pytorch.org/) and [PyG installation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). Other dependencies are included in `requirements.txt` and can be installed via `pip install -r requirements.txt`

---
## Datasets
### TUSZ
The TUSZ dataset is publicly available and can be accessed from https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml after filling out the data request form. We use TUSZ v1.5.2 in this study.
#### TUSZ data preprocessing
First, we resample all EEG signals in TUSZ to 200 Hz. To do so, run:
```
python data/preprocess/resample_tuh.py --raw_edf_dir {dir-to-tusz-edf-files} --save_dir {dir-to-resampled-signals}
```

### DOD-H
The DOD-H dataset is publicly available and can be downloaded from this [repo](https://github.com/Dreem-Organization/dreem-learning-open).

### ICBEB
The ICBEB dataset is publicly available and can be downloaded using this [script](https://github.com/helme/ecg_ptbxl_benchmarking/blob/master/get_datasets.sh) from this [repo](https://github.com/helme/ecg_ptbxl_benchmarking).
#### ICBEB data preprocessing
We will follow this [repo](https://github.com/helme/ecg_ptbxl_benchmarking) to split the ICBEB dataset into train/validation/test sets, downsample the ECGs to 100 Hz, and obtain nine ECG class labels. To do so, run:
```
python data/preprocess/preprocess_icbeb.py --raw_data_dir <raw-icbeb-data-dir> --output_dir <icbeb-data-dir> --sampling_freq 100
```

---
## Model Training
`scripts` folder shows examples to train GraphS4mer on the three datasets. These scripts have been tested on a single NVIDIA A100 GPU and a single NVIDIA TITAN RTX GPU. If you have a GPU with smaller memory, you can decrease the batch size and set `accumulate_grad_batches` to a value > 1. 
### Model training on TUSZ dataset
To train GraphS4mer on the TUSZ dataset, specify `<dir-to-resampled-signals>`, `<preproc-save-dir>`, and `<your-save-dir>` in `scripts/run_tuh.sh`, then run the following:
```
bash ./scripts/run_tuh.sh
```
Note that the first time when you run this script, it will first preprocess the resampled signals by sliding a 60-s window without overlaps and save the 60-s EEG clips and seizure/non-seizure labels in PyG data object in `<preproc-save-dir>`.

### Model training on DOD-H dataset
To train GraphS4mer on the DOD-H dataset, specify `<dir-to-dodh-data>` and `<your-save-dir>` in `scripts/run_dodh.sh`, then run:
```
bash ./scripts/run_dodh.sh
```

### Model training on ICBEB dataset
To train GraphS4mer on the ICBEB dataset, specify `<icbeb-data-dir>` and `<your-save-dir>` in `scripts/run_icbeb.sh`, then run:
```
bash ./scripts/run_icbeb.sh
```
---
## Updates
* 2023-03: Traffic forecasting related experiments have been moved to the branch `traffic`.

---
## Reference
If you use this codebase, or otherwise find our work valuable, please cite:
```

@InProceedings{pmlr-v209-tang23a,
  title = 	 {Modeling Multivariate Biosignals With Graph Neural Networks and Structured State Space Models},
  author =       {Tang, Siyi and Dunnmon, Jared A and Liangqiong, Qu and Saab, Khaled K and Baykaner, Tina and Lee-Messer, Christopher and Rubin, Daniel L},
  booktitle = 	 {Proceedings of the Conference on Health, Inference, and Learning},
  pages = 	 {50--71},
  year = 	 {2023},
  editor = 	 {Mortazavi, Bobak J. and Sarker, Tasmie and Beam, Andrew and Ho, Joyce C.},
  volume = 	 {209},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {22 Jun--24 Jun},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v209/tang23a/tang23a.pdf},
  url = 	 {https://proceedings.mlr.press/v209/tang23a.html},
}
```
