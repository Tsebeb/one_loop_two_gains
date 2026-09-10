import os

import torch
from omegaconf import DictConfig

from dataset.al_dataset_manager import ActiveLearningDatasetManager
from logger.logger_interface import LoggerInterface
from networks.CLmodel import ContinualLearningModelWrapper


def save_network_state_dict(cfg: DictConfig, iteration_id: int, network: ContinualLearningModelWrapper, ):
    if cfg.get("save_model_state_dict", True):
        print("Save Model State Dicts ... ")
        os.makedirs("models", exist_ok=True)
        model_save_dir = os.path.join("models", f"iter_{iteration_id:02d}_model_state_dict.pth")
        torch.save(network.state_dict(), model_save_dir)

def save_initial_state_dict(cfg: DictConfig, network: ContinualLearningModelWrapper, ):
    if cfg.get("save_model_state_dict", True):
        print("Save Model State Dicts ... ")
        os.makedirs("models", exist_ok=True)
        model_save_dir = os.path.join("models", f"initial_model_state_dict.pth")
        torch.save(network.state_dict(), model_save_dir)

def save_annotation_idcs(iteration_id: int, logger: LoggerInterface, al_dataset_manager: ActiveLearningDatasetManager):
    train_idcs, unl_idcs = al_dataset_manager.get_global_indices(iteration_id)
    logger.log_embedding(iteration_id, f"global_train_idcs", train_idcs)
    logger.log_embedding(iteration_id, f"global_unl_idcs", unl_idcs)
