import os
import sys
import time
import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from dataset.al_dataset_manager import ActiveLearningDatasetManager
from dataset.hydra_dataset_factory import get_al_dataset_manager_from_hydra_config
from logger.json_logger import JsonLogger
from networks.CLmodel import ContinualLearningModelWrapper
from pretrain_strategy.base_pretrain_strategy import BasePretrainStrategy
from utilities.file_output_duplicator import FileOutputDuplicator
from utilities.utils import seed_everything, print_hydra_choices, print_sep_header

OmegaConf.register_new_resolver("torch_dtype", lambda name: getattr(torch, name))


@hydra.main(config_path="hydra_config", config_name="pretrain_run_imagewoof", version_base="1.3")
def main_unsupervised_pretrain(cfg: DictConfig) -> int:
    try:
        # Setup Logging
        sys.stdout = FileOutputDuplicator(sys.stdout, 'stdout.txt', 'w')
        sys.stderr = FileOutputDuplicator(sys.stderr, 'stderr.txt', 'w')
        json_logger = JsonLogger(root_folder=os.getcwd())

        print_hydra_choices(cfg)
        if torch.cuda.is_available():
            gpu_id = cfg.get("gpu_id", 0)
            print(f"JOB got GPU ID {gpu_id} ...")
            base_device = torch.device(f"cuda:{gpu_id}")
        else:
            print("[WARNING] Running on CPU as CUDA is not available ...")
            base_device = torch.device("cpu")
        print("JOB Running on ", base_device)

        # set seed for train / val split
        print("Pipeline Init ... ")
        print("Creating Dataset Manager ... ")
        seed_everything(cfg.seed)
        al_dataset_manager: ActiveLearningDatasetManager = get_al_dataset_manager_from_hydra_config(cfg.dataset, cfg.batch_size, cfg.num_workers, cfg.seed, json_logger)
        print("Creating Network ... ")
        seed_everything(cfg.seed)
        network: ContinualLearningModelWrapper = instantiate(cfg.network)(seed=cfg.seed, num_classes=cfg.dataset.num_classes)
        assert isinstance(network, ContinualLearningModelWrapper), "The network needs to implement the ContinualLearningModelWrapper interface."

        print("Creating Pretrain Strategy ... ")
        seed_everything(cfg.seed)
        training_strategy: BasePretrainStrategy = instantiate(cfg.pretrain_strategy)(model=network, epochs=cfg.epochs, logger=json_logger, torch_device=base_device)

        print("Extracting the Datasets")
        factory = al_dataset_manager.dataloader_factory
        train_ds = al_dataset_manager.get_initial_unannotated_training_data()
        train_dl = factory.get_dataloader(train_ds,"train", True)
        test_ds = al_dataset_manager.get_test_data()
        test_dl = factory.get_dataloader(test_ds,"test", False)

        print_sep_header("Unsupervised Pretraining", num_line_breaks_before=1)
        start_time = time.time()
        seed_everything(cfg.seed)
        pretrain_dict = training_strategy.pretrain(train_dl, test_dl)
        try:
            json_logger.log_dictionary(0, "pretraining_stats", pretrain_dict)
        except Exception as e:
            print(e)

        print("Saving States and state dicts ... ")
        training_strategy.save_checkpoint(os.getcwd())
        pretrained_state_dict = network.state_dict()
        hydra_config = HydraConfig.get()
        checkpoint_path = os.path.join(hydra.utils.get_original_cwd(), "checkpoints", f"seed_{cfg.seed}", hydra_config.runtime.choices.dataset, hydra_config.runtime.choices.network)
        os.makedirs(checkpoint_path, exist_ok=True)
        torch.save(pretrained_state_dict, os.path.join(checkpoint_path, f"model_state_dict.pth"))

        print(f"Total Ellapsed Time: {(time.time() - start_time) / 3600.0:.2f} hours ... ")
        print("Pretraining has ended ... ")
        return 0
    except Exception as e:
        import traceback
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        raise e


if __name__ == "__main__":
    main_unsupervised_pretrain()
    sys.exit(0)
