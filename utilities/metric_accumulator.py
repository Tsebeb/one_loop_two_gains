import numpy as np
import torch
import sys


class MetricAccumulator:
    def __init__(self,):
        self.accumulation_dict = {}


    def update_dict(self, dict_to_update: dict):
        for key in dict_to_update.keys():
            if key not in self.accumulation_dict:
                if isinstance(dict_to_update[key], torch.Tensor):
                    self.accumulation_dict[key] = dict_to_update[key].unsqueeze(0)
                else:
                    self.accumulation_dict[key] = dict_to_update[key]
            else:
                if isinstance(dict_to_update[key], (int, str, float, bool)):
                    if isinstance(self.accumulation_dict[key], (int, str, float, bool)):
                        self.accumulation_dict[key] = [self.accumulation_dict[key], dict_to_update[key]]
                    else:
                        self.accumulation_dict[key].append(dict_to_update[key])
                elif isinstance(dict_to_update[key], torch.Tensor):
                    cat_value = torch.unsqueeze(dict_to_update[key], dim=0)
                    self.accumulation_dict[key] = torch.cat((self.accumulation_dict[key], cat_value), dim=0)
                elif isinstance(dict_to_update[key], np.ndarray):
                    self.accumulation_dict[key] = np.concatenate((self.accumulation_dict[key], dict_to_update[key]), axis=0)
                else:
                    print("WARNING unsupported accumulation type for aggregation metrics ...", file=sys.stderr)


    def get_dict_repr(self) -> dict:
        return self.accumulation_dict