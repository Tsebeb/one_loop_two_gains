from typing import Tuple, List
import torch
from torch.nn import Linear
from torchvision.models import ResNet18_Weights
from networks.CLmodel import ContinualLearningModelWrapper
from torchvision.models.resnet import resnet18
from utilities.prune_utils import get_prune_list
import os
import hydra
from utilities.constants import Constants as c
from typing import Optional


class ResNet18(ContinualLearningModelWrapper):
    def __init__(self, num_classes: int, pretrain_imagenet: bool, seed: int, pretrain_path: Optional[str] = None):
        super().__init__(seed)
        assert not pretrain_imagenet or pretrain_path is None, "Either pretrain imagenet or pretrain path can be set not both ... "
        self.num_classes = num_classes
        self.pretrain_imagenet = pretrain_imagenet
        self.num_classes = num_classes
        self.pretrain_path = pretrain_path

        if self.pretrain_imagenet:
            self._internal_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self._internal_model = resnet18(weights=None)

        self.classifier = Linear(in_features=self._internal_model.fc.in_features, out_features=num_classes)
        self._internal_model.fc = torch.nn.Identity()

        if self.pretrain_path is not None:
            orig_path = hydra.utils.get_original_cwd()
            checkpoint_path = os.path.join(orig_path, c.checkpoint_dir, f"seed_{seed}", pretrain_path)
            assert os.path.exists(checkpoint_path), "The pretrain path needs to exist!"
            model_state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))

            for k in list(model_state_dict.keys()):
                if "fc" in k or "classifier" in k:
                    del model_state_dict[k]
            self.load_state_dict(model_state_dict, strict=False)

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
        model = ResNet18(num_classes=100, pretrain_imagenet=False)
        model.eval()
        pred = model(pseudo_input)

        embedding = model.get_embedding(pseudo_input)
        pred2 = model.predict_from_embedding(embedding)
        print("All Close direct and from embeddings", torch.allclose(pred, pred2))
        print("Embeddings dimensions: ", embedding.shape)
        print("Parameters: ", sum(p.numel() for p in model.parameters()))