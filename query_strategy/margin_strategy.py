import torch
from dataset.iteration_dataset_view import IterationDatasetView
from logger.logger_interface import LoggerInterface
from networks.CLmodel import ContinualLearningModelWrapper
from query_strategy.dropout_strategy import DropoutStrategy


class MarginSampling(DropoutStrategy):
    """
    TODO: Add paper and reference implementation
    """
    def __init__(self, model: ContinualLearningModelWrapper, logger: LoggerInterface, torch_device, dropout_iterations: int = 1):
        super(MarginSampling, self).__init__(model, logger, torch_device, dropout_iterations)

    def calculate_single_dropout_score(self, iteration_id: int, iteration_dataset_view: IterationDatasetView) -> torch.Tensor:
        unlabelled_dl = iteration_dataset_view.get_unlabelled_dl(transform_name="test", shuffle=False)

        with torch.inference_mode():
            margin_scores = []
            for x, _ in unlabelled_dl:
                x = x.to(self.torch_device)
                logits = self.model(x)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                topk_probabilities = torch.topk(probabilities, k=2, largest=True, sorted=True, dim=1)
                margin_tensors = 1 - (topk_probabilities.values[:, 0] - topk_probabilities.values[:, 1])  # Margin between samples
                margin_scores.append(margin_tensors.cpu())
        return torch.cat(margin_scores, dim=0)
