# Attacks via Machine Learning: Uncovering Privacy Risks in Sequential Data Releases

## Overview
This is the official repository for Attacks via Machine Learning: Uncovering Privacy Risks in Sequential Data Releases.

## Installation
   ```bash
   pip install -r requirements.txt
   ```

## Trajectory Setting
### 1. Data Preprocessing

You can download the trajectory data from [Geolife](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/). After saving the dataset in `data` folder, you can run the following command:
```bash
python code/trajectory/trajectory_data_processing.py
```

This will create a folder named `processed` under `data/geolife` with the selected trajectories based on geographic range and time length.

### 2. Run Attack

You can run attack using the following command, which will first create published regions for each trajectories according to `lambda_value` and `deviation_amount_user` and then perform attacks using Bi-Directional HMM-RL algorithm and baseline (PI-uniform attack) algorithm given only the generated publish regions:

```bash
python code/trajectory/main.py \
    --dataset <DATASET> \
    --lambda_value <LAMBDA_VALUE> \
    --deviation_amount_user <DEVIATION_AMOUNT_USER> \
    --delta <DELTA> \
    --num_iter <NUM_ITER> \
    --k <K> \
    --gamma <GAMMA>
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
- **--k**
Window size in Bi-Directional HMM-RL algorithm (default = 3).
- **--gamma**
Gamma in Bi-Directional HMM-RL algorithm (default = 5).

An example command is:

```bash
python code/trajectory/main.py \
    --dataset 'geolife' \
    --lambda_value 0.1 \
    --deviation_amount_user 1 \
    --delta 0.7 \
    --num_iter 50 \
    --k 3 \
    --gamma 5
```

Graphs visualizing the predicted and ground truth trajectories will be saved to `graph_output` folder.


## Data Buyers Setting

### 1. Run Attack

The Adults dataset is available in the `data` folder. You can run attack using the following command, which will first create true and published intents according to `lambda_value` and then perform attacks using Bi-Directional HMM-RL algorithm and baseline (PI-uniform attack) algorithm given only the generated publish intents:

```bash
python code/data_market/main.py \
    --total_time <TOTAL_TIME> \
    --lambda_value <LAMBDA_VALUE> \
    --verbose <VERBOSE> \
    --num_iter <NUM_ITER>
```

- **--total_time**
Total time length considered in simulation when generating true intent. (default: 10).
- **--lambda_value**
Maximum confidence threshold determined by buyer (default: 0.5).
- **--verbose**
Verbose or not (default: "False").
- **--num_iter**
Number of iterations in Bi-Directional HMM-RL algorithm (default = 10).

An example command is:

```bash
python code/data_market/main.py \
    --total_time 10 \
    --lambda_value 0.5 \
    --verbose "True" \
    --num_iter 10
```

### 2. Visualization
Using the above command, attacks using different values of hyperparameters ($\beta, \gamma, \delta, k$) will be performed, and the results will be saved to `code/data_market/output/hmm_rl_results.json`. To visualize the predictions by the optimal set of hyperparameters and the effect of varying different hyperparameters, you can use the following command:
```bash
python code/data_market/draw_graph.py
```

Graphs will be saved to `output` folder.