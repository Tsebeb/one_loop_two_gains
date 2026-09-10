import os
import random
from hydra.core.hydra_config import HydraConfig
import numpy as np
import torch
from omegaconf import DictConfig

from dataset.iteration_dataset_view import IterationDatasetView
from training_strategy.base_training import BaseTrainingStrategy


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_hydra_choices(base_cfg: DictConfig):
    hydra_config = HydraConfig.get()
    print_sep_header("Hydra Config")
    print("Seed: ", base_cfg.get("seed", None))
    print("GPU ID: ", base_cfg.get("gpu_id", None))
    print(f"Current Process ID: {os.getpid()}")
    for key, value in hydra_config.runtime.choices.items():
        if not key.startswith("hydra/"):
            print(f"{key}: {value}")
    print_sep_header(num_line_breaks_after=3)


def print_sep_header(text: str = "", length: int = 120, symbol: str = "=", num_line_breaks_before: int = 0, num_line_breaks_after: int = 0):
    if len(text) > 0:
        padded_text = f" {text} "

        num_symbols = (length - len(padded_text))
        if num_symbols % 2 == 0:
            print_text = symbol * (num_symbols // 2) + padded_text + symbol * (num_symbols // 2)
        else:
            print_text = symbol * (num_symbols // 2 + 1) + padded_text + symbol * (num_symbols // 2)
    else:
        print_text = symbol * length
    for i in range(num_line_breaks_before):
        print()
    print(print_text)
    for i in range(num_line_breaks_after):
        print()


def print_log_results(pre_name: str, metrics_dict: dict):
    print(f"{pre_name:15}", end="")
    for key, value in metrics_dict.items():
        if isinstance(value, list):
            continue
        print(f" | {key:8}: {value:8.4f}", end="")
    print()


def evaluate_and_save_metrics(al_dataset_manager, iteration_id: int, training_strategy: BaseTrainingStrategy, json_logger, cfg: DictConfig):
    ds_view: IterationDatasetView = al_dataset_manager.get_dataset_view()
    # Eval on everything
    test_dl = al_dataset_manager.dataloader_factory.get_dataloader(al_dataset_manager.get_test_data(), "test", False)
    if ds_view.get_num_labelled_data() > 0 and cfg.get("evaluate_train", True):
        train_eval, train_embeddings = training_strategy.evaluate(ds_view.get_train_dl(), return_embeddings=True)
        print_log_results("Train", train_eval)
        json_logger.log_dictionary(iteration_id, "evaluation_train_metric", train_eval)
        if cfg.get("save_train_embeddings", False):
            json_logger.log_embedding(iteration_id, "train_embeddings", train_embeddings)

    if ds_view.get_num_unlabelled_data() > 0 and cfg.get("evaluate_unl", True):
        unl_eval, unl_embeddings = training_strategy.evaluate(ds_view.get_unlabelled_dl(), return_embeddings=True)
        print_log_results("Unlabelled", unl_eval)
        json_logger.log_dictionary(iteration_id, "evaluation_unlabelled_metric", unl_eval)
        if cfg.get("save_unl_embeddings", False):
            json_logger.log_embedding(iteration_id, "unlabelled_embeddings", unl_embeddings)

    if ds_view.get_num_val_data() > 0 and cfg.get("evaluate_val", True):
        val_eval, val_embeddings = training_strategy.evaluate(ds_view.get_val_dl(), return_embeddings=True)
        print_log_results("Validation", val_eval)
        json_logger.log_dictionary(iteration_id, "evaluation_validation_metric", val_eval)
        if cfg.get("save_val_embeddings", False):
            json_logger.log_embedding(iteration_id, "validation_embeddings", val_embeddings)

    test_eval, test_embeddings = training_strategy.evaluate(test_dl, return_embeddings=True)
    print_log_results("Test", test_eval)
    json_logger.log_dictionary(iteration_id, "evaluation_test_metric", test_eval)
    if cfg.get("save_test_embeddings", False):
        json_logger.log_embedding(iteration_id, "test_embeddings", test_embeddings)
