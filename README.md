# Attacks via Machine Learning: Uncovering Privacy Risks in Sequential Data Releases

## Overview
This is the official repository for Attacks via Machine Learning: Uncovering Privacy Risks in Sequential Data Releases.

## Installation
   ```bash
   pip install -r requirements.txt
   ```

## 1. Data Preprocessing

To preprocess the trajectory data from [Geolife](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/) (in `data` folder), you can run the following command:
```bash
python trajectory_data_processing.py
```

This will create a folder named `processed` under `data/geolife` with the selected trajectories based on geographic range and time length.

## 2. Run Attack

You can run attack using the following command, which will first create published regions for each trajectories according to `lambda_value` and `deviation_amount_user` and then perform attacks using Bi-Directional HMM-RL algorithm and baseline (PI-uniform attack) algorithm given only the generated publish regions:

```bash
python main.py \
    --dataset <DATASET> \
    --lambda_value <LAMBDA_VALUE> \
    --deviation_amount_user <DEVIATION_AMOUNT_USER> \
    --delta <DELTA> \
    --num_iter <NUM_ITER>
```

- **--dataset**
Dataset to use.
- **--lambda_value**
Maximum confidence threshold determined by user (default: 0.1).
- **--deviation_amount_user**
Amount of deviation set by user when generating PR (default: 0).
- **--delta**
Threshold for reward/penalization (default: 0.3).
- **--num_iter**
Number of iterations in Bi-Directional HMM-RL algorithm (default = 100).

An example command is:

```bash
python main.py \
    --dataset 'geolife' \
    --lambda_value 0.1 \
    --deviation_amount_user 1 \
    --delta 0.7 \
    --num_iter 50
```

Graphs visualizing the predicted and ground truth trajectories will be saved to `graph_output` folder.
