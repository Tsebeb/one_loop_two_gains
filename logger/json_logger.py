import json
import os
from typing import Union

import numpy as np
import torch

from logger.logger_interface import LoggerInterface


class JsonTensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class JsonLogger(LoggerInterface):


    def __init__(self, root_folder: str):
        super(JsonLogger, self).__init__()
        self.root_folder = root_folder
        self.log_name = "exp_log.json"
        self.full_path = os.path.join(self.root_folder, self.log_name)
        self.internal_log_structure = {}

    def __check_or_create_cycle_iteration(self, cycl_iteration: int):
        if cycl_iteration not in self.internal_log_structure:
            self.internal_log_structure[cycl_iteration] = {}

    def __flush_internal_structure(self):
        with open(self.full_path, "w") as f:
            json.dump(self.internal_log_structure, f, cls=JsonTensorEncoder)

    def log_value(self, cycle_iteration: int, value_name: str, value: Union[int, float, list]):
        self.__check_or_create_cycle_iteration(cycle_iteration)
        self.internal_log_structure[cycle_iteration][value_name] = value
        self.__flush_internal_structure()

    def log_embedding(self, cycle_iteration: int, value_name: str, embedding: Union[torch.Tensor, np.ndarray, list]):
        self.__check_or_create_cycle_iteration(cycle_iteration)
        if isinstance(embedding, np.ndarray):
            name_embedding = os.path.join(self.root_folder, f"iter_{cycle_iteration}_{value_name}.npy")
            np.save(name_embedding, embedding)
        elif isinstance(embedding, torch.Tensor):
            name_embedding = os.path.join(self.root_folder, f"iter_{cycle_iteration}_{value_name}.pt")
            torch.save(embedding, name_embedding)
        elif isinstance(embedding, list):
            embedding = np.array(embedding)
            name_embedding = os.path.join(self.root_folder, f"iter_{cycle_iteration}_{value_name}.npy")
            np.save(name_embedding, embedding)
        else:
            raise RuntimeError("Unsupported Log Structure?")

        self.internal_log_structure[cycle_iteration][value_name] = name_embedding
        self.__flush_internal_structure()

    def log_dictionary(self, cycle_iteration: int, value_name: str, dictionary: dict):
        self.__check_or_create_cycle_iteration(cycle_iteration)
        self.internal_log_structure[cycle_iteration][value_name] = dictionary
        self.__flush_internal_structure()

    def log_path(self, cycle_iteration: int, value_name: str, path: str):
        self.__check_or_create_cycle_iteration(cycle_iteration)
        self.internal_log_structure[cycle_iteration][value_name] = path
        self.__flush_internal_structure()

    def __del__(self):
        try:
            self.__flush_internal_structure()
        except:
            pass
