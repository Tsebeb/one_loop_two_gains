from typing import Tuple, List, Union, Iterator, Optional
import torch
from timm import create_model
from timm.models import VisionTransformer, ConvNeXt
from torch.nn import Linear, Parameter
from networks.CLmodel import ContinualLearningModelWrapper
from utilities.prune_utils import get_prune_list, apply_pruning_structure


class TimmBasedBackbone(ContinualLearningModelWrapper):
    def __init__(self, timm_model_name: str, num_classes: int, pretrained: bool, seed: int, train_head_only: bool = False, backbone_lr: Optional[float] = None,
                 vit_cascade: bool = False, vit_cascade_decay_factor: float = 0.8, start_pruning_block: Optional[int] = None):
        assert timm_model_name in ['deit_small_patch16_224', 'vit_small_patch16_224.dino', 'convnextv2_tiny.fcmae'], "Not supported tested yet ... check for yourself ... "
        super().__init__(seed)
        self.timm_model_name = timm_model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.train_head_only = train_head_only
        self.vit_cascade = vit_cascade
        self.vit_cascade_decay_factor = vit_cascade_decay_factor
        self.start_pruning_block = start_pruning_block
        self.backbone_lr = backbone_lr
        self._internal_model: Union[VisionTransformer, ConvNeXt] = create_model(timm_model_name, pretrained=self.pretrained, num_classes=0)  # Remove the final head lyaer with num_classes 0
        if self.train_head_only:
            self._internal_model.requires_grad_(False)  # Block all Updates here ...
        self.classifier = Linear(in_features=self._internal_model.num_features, out_features=num_classes)
        torch.nn.init.zeros_(self.classifier.bias)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        if self.backbone_lr is None:
            return super().parameters()
        else:
            head_params = {"params": self.classifier.parameters()}
            if self.vit_cascade and self.timm_model_name in ['deit_small_patch16_224', 'vit_small_patch16_224.dino']:
                params = [head_params]

                cur_lr_backbone = self.backbone_lr
                for block_idx in range(len(self._internal_model.blocks) -1, 0, -1):
                    params.append({"params": self._internal_model.blocks[block_idx].parameters(), "lr": cur_lr_backbone})
                    cur_lr_backbone *= self.vit_cascade_decay_factor

                stem_params = []
                name_params = []
                for name, param in self._internal_model.named_parameters():
                    if "blocks" in name:
                        continue
                    stem_params.append(param)
                    name_params.append(name)

                params.append({"params": stem_params, "lr": cur_lr_backbone})
                return params
            else:
                backbone_params = {"params": self._internal_model.parameters(), "lr": self.backbone_lr}
                return [backbone_params, head_params]

    def freeze_backbone(self):
        self._internal_model.requires_grad_(False)

    def unfreeze_backbone(self):
        if not self.train_head_only: # If we are training the head only, we don't want to unfreeze the backbone under any circumstances ... '
            self._internal_model.requires_grad_(True)

    def get_embedding_dim(self) -> int:
        return self.classifier.in_features

    def get_named_backbone_parameters(self) -> List[Tuple[torch.nn.Module, str]]:
        if self.start_pruning_block is not None:
            assert self.timm_model_name in ["deit_small_patch16_224", "vit_small_patch16_224.dino"]
            assert self.start_pruning_block < len(self._internal_model.blocks)
            return get_prune_list(self._internal_model.blocks[self.start_pruning_block:])
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
    pseudo_input = torch.rand((2, 3, 224, 224))
    pseudo_target = torch.zeros((2, 100))
    convnext_name = "convnextv2_tiny.fcmae"
    deit_name = "deit_small_patch16_224"
    dino_name = "vit_small_patch16_224.dino"
    model = TimmBasedBackbone(timm_model_name=convnext_name, num_classes=100, pretrained=False, seed=0, train_head_only=True)
    model.train()
    apply_pruning_structure(model.get_named_backbone_parameters())
    torch.autograd.anomaly_mode.set_detect_anomaly(True)
    opt_list = model.parameters()
    test_opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    with torch.autograd.detect_anomaly():
        for i in range(10):
            pred = model(pseudo_input)
            loss = torch.nn.functional.mse_loss(pred, pseudo_target)
            test_opt.zero_grad()
            loss.backward()
            test_opt.step()  # print("Loss", loss.item())

    with torch.no_grad():
        model.eval()
        pred = model(pseudo_input)

        embedding = model.get_embedding(pseudo_input)
        pred2 = model.predict_from_embedding(embedding)
        print("All Close direct and from embeddings", torch.allclose(pred, pred2))
        print("Embeddings dimensions: ", embedding.shape)
        print("Parameters: ", sum(p.numel() for p in model.parameters()))
