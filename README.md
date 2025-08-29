# multivariate_hawkes_lob

This repository contains the code used to collect Limit Order Book (LOB) data from Bitfinex, convert them into a Pandas dataframe, and use them to train and simulate Multivariate Hawkes Processes.

## Requirements

You need to have **Conda** or **Miniconda** installed on your system.  
If you don’t have it yet, follow the official installation guide:  
https://docs.conda.io/projects/conda/en/latest/user-guide/install/

## Setup

You will create **two conda environments** — one for data collection and one for training/simulation.

### 1) Environment for collecting LOB data
Create and activate the environment, then install the Python dependencies from `requirements_bitfinex.txt` (a pip requirements file):

```bash
conda create -n lob_data_env python=3.10 -y
conda activate lob_data_env
python -m pip install -r requirements_bitfinex.txt
```

### 2) Environment for training & simulating Multivariate Hawkes Processes
Create the environment from `environment_hawkes.yml` and force the name to `hawkes_env` for consistency:

```bash
conda env create -n hawkes_env -f environment_hawkes.yml
```

---

## Usage

### A) Collect Limit Order Book data
Activate the data environment and run the collection script. Then convert the raw data to a Pandas dataframe.

```bash
conda activate lob_data_env
python collect_lob_data.py
python convert_lob_data.py
```

### B) Train Multivariate Hawkes Processes
Activate the Hawkes environment and run the training script (greedy algorithm):

```bash
conda activate hawkes_env
python train_multivariate_hawkes_with_greedy_b_all_training_periods_bi.py
```

### C) Simulate events
Still in the Hawkes environment, simulate events:

```bash
python predict_mid_price_events.py
```

---

## Quickstart (copy/paste)

```bash
# 0) Install Conda/Miniconda first (see link above)

# 1) Create LOB data env and install requirements
conda create -n lob_data_env python=3.10 -y
conda activate lob_data_env
python -m pip install -r requirements_bitfinex.txt

# 2) Collect and convert data
python collect_lob_data.py
python convert_lob_data.py

# 3) Create Hawkes env from YAML
conda env create -n hawkes_env -f environment_hawkes.yml
conda activate hawkes_env

# 4) Train
python train_multivariate_hawkes_with_greedy_b_all_training_periods_bi.py

# 5) Simulate
python predict_mid_price_events.py
```

---

## Notes
- The first environment uses a **pip** requirements file; that’s why we use `python -m pip install -r requirements_bitfinex.txt` after creating/activating the environment.
- The second environment is built from a **conda YAML** file; `conda env create -n hawkes_env -f environment_hawkes.yml` ensures the environment is named consistently as `hawkes_env`.

