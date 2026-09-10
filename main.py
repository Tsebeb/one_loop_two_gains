import os
import sys
import time
from typing import Optional
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import hydra
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from dataset.al_dataset_manager import ActiveLearningDatasetManager
from dataset.hydra_dataset_factory import get_al_dataset_manager_from_hydra_config
from initial_data_selection.selection_strategy import StartSelectionStrategy
from logger.json_logger import JsonLogger
from networks.CLmodel import ContinualLearningModelWrapper
from query_strategy.query_strategy import QueryStrategyInterface
from scenario_setting.scenario import BaseScenario
from training_strategy.base_training import BaseTrainingStrategy
from utilities.file_output_duplicator import FileOutputDuplicator
from utilities.save_utils import save_initial_state_dict, save_annotation_idcs, save_network_state_dict
from utilities.utils import seed_everything, evaluate_and_save_metrics, print_sep_header, print_hydra_choices

OmegaConf.register_new_resolver("torch_dtype", lambda name: getattr(torch, name))


@hydra.main(config_path="hydra_config", config_name="train_cifar_lb", version_base="1.3")
def main(cfg: DictConfig) -> int:
    try:
        # Setup Logging
        sys.stdout = FileOutputDuplicator(sys.stdout, 'stdout.txt', 'w')
        sys.stderr = FileOutputDuplicator(sys.stderr, 'stderr.txt', 'w')
        json_logger = JsonLogger(root_folder=os.getcwd())

        json_logger.log_value(0, "success", 0)
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
        al_dataset_manager: ActiveLearningDatasetManager = get_al_dataset_manager_from_hydra_config(cfg.dataset, cfg.batch_size, cfg.num_workers, cfg.seed,
                                                                                                    json_logger,
                                                                                                    validate_on_test=cfg.get("validate_on_test", False))
        print("Creating Network ... ")
        seed_everything(cfg.seed)
        network: ContinualLearningModelWrapper = instantiate(cfg.network)(num_classes=cfg.dataset.num_classes, seed=cfg.seed)
        assert isinstance(network, ContinualLearningModelWrapper), "The network needs to implement the ContinualLearningModelWrapper interface."
        save_initial_state_dict(cfg, network)

        print("Creating Optimizer ... ")
        optimizer: Optimizer = instantiate(cfg.optimizer)(params=network.parameters())
        assert isinstance(optimizer, Optimizer), "The optimizer needs to be a pytorch optimizer"
        lr_scheduler: Optional[LRScheduler] = None
        if "lr_scheduler" in cfg:
            print("Creating LR Scheduler ... ")
            lr_scheduler = instantiate(cfg.lr_scheduler)(optimizer)
            assert isinstance(lr_scheduler, LRScheduler), "The provided lr_scheduler needs to be a pytorch lr scheduler"

        print("Creating Training Strategy ... ")
        seed_everything(cfg.seed)
        training_strategy: BaseTrainingStrategy = instantiate(cfg.training_strategy)(model=network, optimizer=optimizer, epochs=cfg.epochs, lr_scheduler=lr_scheduler, logger=json_logger, torch_device=base_device)

        print("Selecting Initial Data ... ", end="")
        start_time = time.time()
        seed_everything(cfg.seed)
        unlabelled_ds = al_dataset_manager.get_initial_unannotated_training_data()
        initial_selection_strategy: StartSelectionStrategy = instantiate(cfg.initial_selection_strategy)(network=network, torch_device=base_device)
        selected_initial_idcs = initial_selection_strategy.select(unlabelled_ds, al_dataset_manager.dataloader_factory)
        al_dataset_manager.annotate_indices(selected_initial_idcs)
        json_logger.log_embedding(0, "initial_idcs", selected_initial_idcs)
        initial_ds_view = al_dataset_manager.get_dataset_view()
        seed_everything(cfg.seed)
        print("Creating Scenario AL setting... ")
        scenario: BaseScenario = instantiate(cfg.scenario)(al_dataset_manager=al_dataset_manager)
        seed_everything(cfg.seed)
        print("Creating AL Query / Acquisition Strategy ...")
        query_strategy: QueryStrategyInterface = instantiate(cfg.query_strategy)(model=network, logger=json_logger, torch_device=base_device)

        print_sep_header("Initial Split Training", num_line_breaks_before=1)
        seed_everything(cfg.seed)
        training_strategy.train(initial_ds_view)
        evaluate_and_save_metrics(al_dataset_manager, 0, training_strategy, json_logger, cfg)
        save_network_state_dict(cfg, 0, network)
        save_annotation_idcs(0, json_logger, al_dataset_manager)

        seed_everything(cfg.seed)
        while not scenario.has_stream_ended():
            start_iteration_time = time.time()
            scenario.progress_stream()
            iteration_id = scenario.get_current_iteration_id()
            initial_ds_view = al_dataset_manager.get_dataset_view()
            print_sep_header(f"ITERATION {iteration_id:02d}", num_line_breaks_before=3)
            annotation_budget = scenario.get_current_annotation_budget()

            if annotation_budget > 0:
                print("Annotation Budget:", annotation_budget)
                print("Before Annotation Training Size: ", initial_ds_view.get_num_labelled_data())
                print("Before Annotation Unlabelled Size: ", initial_ds_view.get_num_unlabelled_data())
                print("Before Annotation Validation Size: ", initial_ds_view.get_num_val_data())
                print_sep_header()
                print("Calculating AL scores ... ")
                indices_annotation = query_strategy.query_al_pool(iteration_id, initial_ds_view, annotation_budget)
                json_logger.log_embedding(iteration_id, "acquisition_idcs_non_global", indices_annotation)
                al_dataset_manager.annotate_indices(indices_annotation)

                # Training phase
                print("After Annotation ... ")
                updated_ds_view = al_dataset_manager.get_dataset_view()
                print("Training Size: ", updated_ds_view.get_num_labelled_data())
                print("Unlabelled Size: ", updated_ds_view.get_num_unlabelled_data())
                print("Validation Size: ", updated_ds_view.get_num_val_data())
            else:
                updated_ds_view = initial_ds_view

            print_sep_header(f"Training {iteration_id:02d} AL Cycle")
            seed_everything(cfg.seed)
            training_strategy.train(updated_ds_view)

            print_sep_header(f"Evaluation {iteration_id:02d} AL Cycle")
            evaluate_and_save_metrics(al_dataset_manager, iteration_id, training_strategy, json_logger, cfg)
            save_network_state_dict(cfg, iteration_id, network)
            if annotation_budget > 0:
                save_annotation_idcs(iteration_id, json_logger, al_dataset_manager)
            print(f"Ellapsed Time AL Iteration {iteration_id}: {(time.time() - start_iteration_time) / 60.0:.2f} minutes ... ")
            print_sep_header()

        print(f"Total Ellapsed Time: {(time.time() - start_time) / 3600.0:.2f} hours ... ")
        print("Scenario has ended ... ")
        json_logger.log_value(0, "success", 1)
        return 0
    except Exception as e:
        import traceback
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        raise e


if __name__ == '__main__':
    return_code = main()
    sys.exit(return_code)
