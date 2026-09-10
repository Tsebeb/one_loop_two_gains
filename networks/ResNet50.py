from typing import Tuple, List
import torch
from torch.nn import Linear
from torchvision.models import ResNet50_Weights
from networks.CLmodel import ContinualLearningModelWrapper
from torchvision.models.resnet import resnet50
from utilities.prune_utils import get_prune_list


class ResNet50(ContinualLearningModelWrapper):
    def __init__(self, num_classes: int, pretrain_imagenet: bool, seed: int):
        super().__init__(seed)
        self.num_classes = num_classes
        self.pretrain_imagenet = pretrain_imagenet
        self.num_classes = num_classes
        if self.pretrain_imagenet:
            self._internal_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self._internal_model = resnet50(weights=None)
        self.classifier = Linear(in_features=self._internal_model.fc.in_features, out_features=num_classes)
        self._internal_model.fc = torch.nn.Identity()

    def freeze_backbone(self):
        self._internal_model.requires_grad_(False)

    def unfreeze_backbone(self):
        self._internal_model.requires_grad_(True)

    def get_embedding_dim(self) -> int:
        return self.classifier.in_features

    def get_named_backbone_parameters(self) -> List[Tuple[torch.nn.Module, str]]:
        return get_prune_list(self._internal_model)

    def get_num_classes(self) -> int:
        return self.num_classes

    def get_embedding(self, x) -> torch.Tensor:
        return self._internal_model(x)

    def get_embedding_and_logits(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = self._internal_model(x)
        logits = self.classifier(embedding)
        return embedding, logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self._internal_model(x)
        return self.classifier(embedding)

    def predict_from_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.classifier(embedding)


if __name__ == "__main__":
    with torch.no_grad():
        pseudo_input = torch.rand((1, 3, 224, 224))
        model = ResNet50(num_classes=100, pretrain_imagenet=False, seed=0)
        model.eval()
        pred = model(pseudo_input)

        embedding = model.get_embedding(pseudo_input)
        pred2 = model.predict_from_embedding(embedding)
        print("All Close direct and from embeddings", torch.allclose(pred, pred2))
        print("Embeddings dimensions: ", embedding.shape)
        print("Parameters: ", sum(p.numel() for p in model.parameters()))
