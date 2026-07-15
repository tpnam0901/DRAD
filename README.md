<h1 align="center">
  Decentralized relational anomaly detection for electric vehicle batteries through independent learning
  <br>
</h1>

<h4 align="center">Official code repository for the paper "Decentralized relational anomaly detection for electric vehicle batteries through independent learning".</h4>

<p align="center">
<a href=""><img src="https://img.shields.io/github/stars/tpnam0901/DRAD?" alt="stars"></a>
<a href=""><img src="https://img.shields.io/github/forks/tpnam0901/DRAD?" alt="forks"></a>
<a href=""><img src="https://img.shields.io/github/license/tpnam0901/DRAD?" alt="license"></a>
</p>
<div align="center">

[![python](https://img.shields.io/badge/-Python_3.10.14-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![pytorch](https://img.shields.io/badge/Torch_1.13.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![cuda](https://img.shields.io/badge/-CUDA_11.8-green?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit-archive)

</div>

<p align="center">
  <a href="#how-to-use">How To Use</a> •
  <a href="#structure-of-the-project">Structure Of The Project</a> •
  <a href="#license">License</a> •
  <a href="#citation">Citation</a> •
  <a href="#references">References</a> •
</p>

## How To Use

#### Dependencies

- OS Requirements:
  - Linux (Ubuntu/Debian)
  - CUDA >=11.5
  - cuDNN 8.2.4
  - Python >= 3.8
  - PyTorch >=1.13.1

- Our environment:
  - OS: Ubuntu 20.04
  - GPU: NVIDIA 3080ti
  - CUDA 11.5
  - Python 3.10.14
  - PyTorch 1.13.1
- Clone this repository

```bash
git clone https://github.com/tpnam0901/DRAD.git
cd DRAD
```

- Create a conda environment and install requirements

```bash
conda create -n DRAD python=3.10.14 -y
conda activate DRAD
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.5 -c pytorch -c nvidia
pip install -r requirements.txt
```

or with conda environment

```bash
conda env create -f environment.yml
```

- Download the dataset from [here](https://doi.org/10.6084/m9.figshare.23659323)

### Training & Evaluation

#### Dataset Preparation

- Move to data folder [data](./DRAD/data/):

```bash
cd DRAD/data
```

- Unzip the dataset and put the unzipped folder in the data folder. The structure of the dataset in the [data](./DRAD/data/) folder should be as follows:

```data
├── data
│   ├── battery_data
│   │   ├── battery_brand1 # Dahu dataset
│   │   ├── battery_brand2 # Socea dataset
│   │   ├── battery_brand3 # Naobop dataset
```

- Preprocess the dataset. These scripts will merge all the charging sessions for each EV into a single file and generate the metadata for training and evaluation.

```bash
python build_label.py
python preprocess_brand3.py # For Naobop dataset
python preprocess_brand2.py # For Socea dataset
python preprocess_brand1.py # For Dahu dataset
python preprocess_brand1_drift.py # For Dahu dataset with drift (for cross brand evaluation)
```

#### DRAD

- Move to the [DRAD](./DRAD) engine folder:

```bash
cd DRAD
```

- Training with same brand EVs:

```bash
# Modify the path to the dataset metadata in the `run()` function of the evaluation engine before running the code. TODO: move it to the configuration file.
python main.py -m DRV -e t --config configs/DRV.py
                  FL : Federal Learning
                  CL : Centralized Learning
```

- Evaluation with same brand EVs:

```bash
python main.py -m DRV -e e --config configs/DRV.py -cfg_ckpt checkpoints/DRV_DRV_brand3_2025_20260518_190012/config.json
                  FL : Federal Learning
                  CL : Centralized Learning
```

- Training with cross brand EVs:
  - Pre-train the model on Dahu dataset:

  ```bash
  # Modify the path to the dataset metadata in the `run()` function of the evaluation engine before running the code.
  python main.py -m CL -e t --config configs/DRV.py # Modify the configuration file to use Dahu dataset for pre-training
  ```

  - Fine-tune the model on the target dataset (for Socea and Naobop datasets):

  ```bash
  # Modify the path to the dataset metadata in the `run()` function of the evaluation engine before running the code.
  python main.py -m DRV -e t --config configs/DRV_shift.py # Modify the self.pretrained_model_path with the path to the pre-trained model checkpoint. (all_normal_latest.pth)
  ```

- Evaluation with cross brand EVs:

```bash
# cfg_ckpt: Naobop dataset checkpoint for evaluation
# cfg_ckpt2: Socea dataset checkpoint for evaluation
# cfg_ref: Dahu dataset checkpoint for reference (pre-training)
python main.py -m DRV -e e --config configs/DRV_shift.py -cfg_ckpt checkpoints/DRV_DRVShift_brand3_20260422_143848/config.json -cfg_ckpt2 checkpoints/DRV_DRVShift_brand2_20260422_143231/config.json -cfg_ref checkpoints/CL_DRV_brand1_20260415_141406/config.json
```

- All the checkpoints of DRAD are deposited in the [checkpoints](https://github.com/tpnam0901/DRAD/releases).

#### Other Baselines

#### Dataset Preparation

- Create a symbolic link to the dataset in the [data](./DRAD/data/) folder:

```bash
cd Others
mkdir -p working/dataset
cd working/dataset
ln -s ../../../DRAD/data/data/battery_data RFDBattery
```

- Start training:

```bash
python main.py -m CL -e t --config configs/BaselineName.py
```

- Start evaluation:

```bash
python main.py -m CL -e e --config configs/BaselineName.py -cfg_ckpt working/checkpoints/BaselineName_checkpoint/config.json
                  DRV
```

## Structure Of The Project

- The overview of the project structure is as follows:

```
├── configs
│   ├── all the configuration files for training and evaluation
│   └── ...
├── data
│   ├── all scripts used for data loading, preprocessing, and augmentation
│   └── ...
├── engine
│   ├── all the training and evaluation loops, including the main logic for model optimization and performance evaluation.
│   ├── this module will also include the implementation of the DRAD framework and other baseline models for comparison.
│   └── ...
├── mlruns
│   ├── all the logs generated during training and evaluation
│   └── ...
├── networks
│   ├── all the deep learning models used in the project
│   └── ...
├── utils
│   ├── all utility functions and helper scripts for training and evaluation such as metrics, visualization, loss functions, etc.
│   └── ...
├── main.py (Starting point for using the codebase)
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
└── ...
```

- How it works:
  - main.py reads the configuration file from the [configs](./DRAD/configs).
  - Based on the configuration, corresponding engine will be initialized and `run()` function will be called to start the training and evaluation process.
  - Run function will first initialize the dataset. Note: the path to the dataset metadata need to modified in the run function before running the code. TODO: move it to the configuration file.
  - To add new engine/baseline, add the implementation in the [engine](./DRAD/engine) module and add the corresponding configuration file in the [configs](./DRAD/configs) folder. Then, modify the main.py to include the new engine/baseline. The new engine/baseline can be inherited from the base engine and only need to implement the specific logic for training and evaluation.
  - `get_groups()` function in the engine module need to modified based on the target evaluation metric. TODO: move it to the configuration file.
- Main implementation of the DRAD framework is in the [evaluate_drv.py](./DRAD/engine/evaluate_drv.py) and [evaluate_drv_cross.py](./DRAD/engine/evaluate_drv_cross.py) files.
  - Step 1: get all normal and abnormal EV ids from the dataset.
  - Step 2: get the groups of EVs based on the target evaluation metric along with its corresponding normal and abnormal EV ids.
  - Step 3: find per-error EVs for each group using `eval_group()` function. This function will return the per-error EVs for each group.
  - Step 4: find outliers through statistical analysis of the per-error EVs. Function `find_outlier_with_mean_std()`.
  - Step 5: calculate the evaluation metrics based on the identified outliers and the ground truth labels.
  - Step 6: back to step 4 with different outlier detection method (`find_outlier_with_IQR() and find_outlier_with_group()`).
  - Step 7: calculate the average performance across all groups and the overall performance across all EVs.
  - Note: function `plot_error_matrix()` can be used to visualize the error matrix for each group. Comment line `225 confusion_matrix[j, i] = each_car_errors[car_id][car_data_id]` for remove diagonal normalization in the error matrix.
  - Note: alpha and beta values will be adjusted in the `__init__()` function of the evaluation engine. TODO: move it to the configuration file.

## References

[1] Zhang, J., Wang, Y., Jiang, B., He, H., Huang, S., Wang, C., ... & Ouyang, M. (2023). Realistic fault detection of li-ion battery via dynamical deep learning. Nature Communications, 14(1), 5940.

---

> GitHub [@tpnam0901](https://github.com/tpnam0901) &nbsp;&middot;&nbsp;
