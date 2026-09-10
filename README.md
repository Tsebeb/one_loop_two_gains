# Reference Implementation for ___One Loop Two Gains: Can Active Learning win the Lottery for Free?___

Source code accompanying the paper *One Loop Two Gains: Can Active Learning win the Lottery for Free?* that has been accpeted at the
fifth conference on Lifelong Learning Agents - CoLLAs 2026 - in Bucharest, Romania. This repository provides a modular, Hydra-based framework for reproducing all experiments presented in the paper.

---

## Project Structure

```
.
├── main.py                         # Main entry point (AL training loop)
├── main_unsupervised_pretrain.py   # SimCLR pretraining entry point
├── hydra_config/                   # All experiment configurations (Hydra YAML)
│   ├── dataset/                    # Dataset definitions
│   ├── network/                    # Network architectures
│   ├── training_strategy/          # Training & pruning strategies
│   ├── query_strategy/             # AL acquisition functions
│   ├── scenario/                   # AL budget & cycle settings
│   └── initial_selection_strategy/ # Initial labeled pool selection
├── dataset/                        # Dataset management & AL pool logic
├── networks/                       # Model implementations
├── query_strategy/                 # Acquisition strategy implementations
├── training_strategy/              # Training loop & pruning logic
├── pretrain_strategy/              # SimCLR pretraining
├── scenario_setting/               # Pool-based AL scenario
├── initial_data_selection/         # Initial pool selection strategies
├── utilities/                      # Pruning, reactivation & helper utilities
└── logger/                         # JSON-based metric logging
```

## Environment Setup

Install all required dependencies:

```bash
pip install -r requirements.txt
```

Experiments were conducted with **PyTorch 2.9.0** and **torchvision 0.24.0**.
A full snapshot of the environment used on our test machine is provided in `framework_environment.yml`.

## Reproducing Results

Reproduction follows three stages: (1) self-supervised pretraining, (2) active learning runs, and (3) inspecting results.

### 1. Self-Supervised Pretraining (SimCLR)

CIFAR-100, ImageWoof, and TinyImageNet experiments use SimCLR-pretrained ResNet-18 backbones.
Pretraining must be completed before running the corresponding AL experiments.

```bash
python main_unsupervised_pretrain.py --config-name pretrain_run_cifar
python main_unsupervised_pretrain.py --config-name pretrain_run_imagewoof
python main_unsupervised_pretrain.py --config-name pretrain_run_tiny_imagenet
```

Pretrained weights are saved to `checkpoints/seed_<seed>/<dataset>/<network>/model_state_dict.pth` and are loaded automatically by the downstream training configs.
To run pretraining across multiple seeds, append `seed=0,1,2,3,4` with `--multirun`. By default seed 0 is trained. 
### 2. Active Learning Training Runs

Each top-level config in `hydra_config/` defines a complete experiment.
Configs are set up as **multirun sweeps** by default, iterating over all training strategies and query strategies reported in the paper
for a single seed 0. If checkpoint is not available an AssertionError will be triggered.


```bash
python main.py --config-name train_cifar_hb
python main.py --config-name train_cifar_lb
python main.py --config-name train_woof_hb
python main.py --config-name train_woof_lb
python main.py --config-name train_tiny
python main.py --config-name train_places_vit
python main.py --config-name train_places_cnn
```

> **Note:** Places365 requires a local copy of the dataset with the path configured in `hydra_config/dataset/places365_small_no_val_cluster.yaml`. All other datasets download automatically to `data/`.

### 3. Results

Results are written to the `results/` directory, organized by dataset, seed, query strategy, and training strategy. Each run produces JSON metric files (e.g., test accuracy per AL iteration) alongside the full Hydra config used.

```
results/<dataset>/<seed>/<query_strategy>/<training_strategy>/
```
by defaul results are saved into according "exp_log.json" files containing all metadata and indices. The utilities for the flops_calculation can be found under utilities.flop_utils.


## Citation

If you find the papers work interesting or the framework source code helpful please consider citing us by

    Coming Soon