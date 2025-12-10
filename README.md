# QNI Experiments: Fairness & Membership

This repository contains the full codebase used to run the Quantitative Non-Interference (QNI) experiments.  A separate written report complements this project and explains:
- the theoretical background,
- the experimental design,
- and the analysis of the results.

This README documents the code, how it is structured, and how to run the experiments to verify the results.

## Project structure

- `config.py`  
  Defines two dataclasses:
  - `TrainConfig`: training hyperparameters (model type, batch size, lr, DP settings, device, etc.).
  - `ExperimentConfig`: experiment setup (task, dataset, protected attribute, output directories).

- `datasets.py`  
  - Downloads or reuses cached CSVs for **Adult** and **COMPAS** via `kagglehub`, stores them under `./data/`.
  - Encodes features, labels, and a multi-class protected attribute (as the last feature column).
  - Wraps them in a `TabularDataset` (returns batches as `{"x", "y", "protected"}`).
  - Provides `get_mnist_dataloaders` for MNIST (train/test loaders).
  - Saves metadata about protected attribute encodings to JSON for later plotting.

- `adjacency.py`  
  - `make_fairness_adjacent_batch(...)`: given a batch with a protected attribute, creates an adjacent batch where only the protected attribute is rotated to the next class, all other features bleiben unchanged.
  - `split_for_membership(...)`: creates two datasets that differ in exactly one record.

- `models.py`  
  - `LogisticRegression` for fairness data.
  -  `SimpleMLP` for MNIST.
 
- `training.py`  
  - `train_standard(...)`: standard training .
  - `train_dp(...)`: DP-SGD training.
  - `collect_outputs(...)`: utility to collect logits and labels from a loader.

- `dp_sgd.py`  
  - `SimpleGaussianAccountant`: pessimistic privacy accountant using a Gaussian mechanism composition bound.
  - `dp_sgd_step(...)`: one DP-SGD update on a batch (per-example grads, clipping, noise, optional accounting).

- `stats.py`  
  - `softmax(...)`: row-wise softmax.
  - `ks_test(...)`: two-sample Kolmogorov–Smirnov test.
  - `welch_ttest(...)`: Welch’s t-test (unequal variances).
  - `total_variation_distance(...)`: average TV distance between output distributions.
  - `empirical_epsilon(...)`: heuristic empirical ε from log probability ratios.
  - `accuracy_from_probs(...)`: classification accuracy from probability outputs.

- `experiments_fairness.py`  
  - Loads Adult or COMPAS, builds/load model, and runs fairness QNI evaluation.
  - Constructs original vs. adjacent batches using `make_fairness_adjacent_batch`.
  - Computes global metrics: accuracy, KS, Welch, TV distance, empirical ε.
  - Saves:
    - Text summary: `results/fairness_<dataset>_<protected>_dp<0|1>.txt`
    - JSON summary: `results/fairness_<dataset>_<protected>_dp<0|1>.json`

- `experiments_membership.py`  
  - Runs a membership experiment on MNIST:
    - Train `model_with` on full data and `model_without` with one record removed.
    - Compare predictive distributions on the test set (KS, Welch, TV, empirical ε).
  - Saves:
    - Text summary: `results/membership_mnist_dp<0|1>.txt`
    - JSON summary: `results/membership_mnist_dp<0|1>.json`

- `main.py` 
  - Parses CLI arguments:
    - `--task {fairness,membership}`
    - `--dataset {adult,compas,mnist}`
    - `--dp` (flag to enable DP-SGD)
  - Builds `TrainConfig` and `ExperimentConfig`.
  - Dispatches to `run_fairness_experiment` or `run_membership_experiment`.

## How to Run the Experiments

### Install Requirements

pip install -r requirements.txt

### Run (`main.py`)

`main.py` parses the following arguments:

#### `--task {fairness,membership}`
Selects which experiment to run.

#### `--dataset {adult,compas,mnist}`
- **Fairness:** `adult` or `compas`
- **Membership:** only `mnist` is used (other values are ignored)

### `--protected_attr {gender,race}`
- Used **only for fairness experiments**
- Ignored for membership experiments

#### `--dp`
Training method selector:
- **DP-SGD:** --dp present
-  **standard SGD:** --dp not present


Example run: python main.py --task fairness --dataset adult --protected_attr race --

## Outputs

Running any experiment automatically creates:

### results/

For fairness:

fairness_<dataset>_<protected_attr>_dp<0|1>.txt

fairness_<dataset>_<protected_attr>_dp<0|1>.json

For membership:

membership_mnist_dp<0|1>.txt

membership_mnist_dp<0|1>.json

The .txt files provide readable summaries.
The .json files contain the same metrics in a structured form for plotting.

### models/

Contains trained model checkpoints :

<dataset>_<protected_attr>_<model_type>_dp<0|1>.pt

### data/

Contains:

Downloaded datasets and protected-attribute encoding files
