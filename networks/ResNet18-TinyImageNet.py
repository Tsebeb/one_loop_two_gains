import os.path
import sys
from typing import Tuple, List, Optional
import hydra.utils
import torch
from networks.CLmodel import ContinualLearningModelWrapper
from networks.MLP import MLP
from utilities.prune_utils import get_prune_list
from utilities.constants import Constants as c


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.act1 = torch.nn.ReLU(inplace=False)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.act2 = torch.nn.ReLU(inplace=False)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act2(out)
        return out


class ResNet18TinyImage(ContinualLearningModelWrapper):
    def __init__(self, seed: int, num_classes=10, dropout: float = 0.0, hidden_layer: Optional[List[int]] = None, hidden_activation_name="swish", pretrain_path: Optional[str] = None):
        super(ResNet18TinyImage, self).__init__(seed)
        self.in_planes = 64
        self.num_classes = num_classes

        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.act1 = torch.nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.avgpool = torch.nn.AvgPool2d(4)
        self.flatten = torch.nn.Flatten()
        self.head = MLP(input_dimension=512 * BasicBlock.expansion, layer_sizes=(hidden_layer if hidden_layer is not None else []) + [num_classes],
                        output_activation_name="identity", hidden_activation_name=hidden_activation_name, dropout_factor=dropout)

        if pretrain_path is not None:
            orig_path = hydra.utils.get_original_cwd()
            checkpoint_path = os.path.join(orig_path, c.checkpoint_dir, f"seed_{seed}", pretrain_path)
            assert os.path.exists(checkpoint_path), "The pretrain path needs to exist!"
            model_state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))

            for k in list(model_state_dict.keys()):
                if "head" in k:
                    del model_state_dict[k]
            self.load_state_dict(model_state_dict, strict=False)

    def freeze_backbone(self):
        self._internal_model.requires_grad_(False)

    def unfreeze_backbone(self):
        self._internal_model.requires_grad_(True)

    def get_embedding_dim(self) -> int:
        return 512

    def get_named_backbone_parameters(self) -> List[Tuple[torch.nn.Module, str]]:
        return get_prune_list(self.conv1, self.layer1, self.layer2, self.layer3, self.layer4)

    def get_num_classes(self) -> int:
        return self.num_classes

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        embedding = self.get_embedding(x)
        out = self.head(embedding)
        return out

    def get_embedding(self, x) -> torch.Tensor:
        out = self.act1(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = self.avgpool(out4)
        embedding = self.flatten(out)
        return embedding

    def get_embedding_and_logits(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = self.get_embedding(x)
        logits = self.head(embedding)
        return embedding, logits

    def predict_from_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.head(embedding)


if __name__ == "__main__":
    pseudo_input = torch.zeros((1, 3, 64, 64))
    model = ResNet18TinyImage(num_classes=100, seed=0)
    with torch.profiler.profile(with_flops=True) as prof:
        with torch.no_grad():
            prediction = model(pseudo_input)
    total_flops = sum([e.flops for e in prof.key_averages() if e.flops is not None])
    print("TOTAL FLOPS:", total_flops)
    sys.exit(0)
