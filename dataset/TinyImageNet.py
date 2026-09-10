import os
from glob import glob

import torchvision.tv_tensors
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2.functional import pil_to_tensor

from dataset.download_utils import download_with_progress, extract_zip_with_progress


class TinyImageNetDataset(Dataset):
    """
    PyTorch dataset for Tiny ImageNet (200 classes, 64x64).
    Reads the native extracted structure without any reorganization.
      - train: tiny-imagenet-200/train/<wnid>/images/*.JPEG
      - val:   tiny-imagenet-200/val/images/*.JPEG + val_annotations.txt
    """
    def __init__(self, ds_path: str, is_train: bool = False):
        super().__init__()
        root = os.path.join(ds_path, "tiny-imagenet-200")

        # Build wnid -> class index mapping from wnids.txt
        with open(os.path.join(root, "wnids.txt"), "r") as f:
            wnids = [line.strip() for line in f if line.strip()]
        self._wnid_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}

        self.file_paths = []
        self.targets = []

        if is_train:
            train_dir = os.path.join(root, "train")
            for wnid in wnids:
                class_idx = self._wnid_to_idx[wnid]
                img_dir = os.path.join(train_dir, wnid, "images")
                for img_path in sorted(glob(os.path.join(img_dir, "*.JPEG"))):
                    self.file_paths.append(img_path)
                    self.targets.append(class_idx)
        else:
            val_dir = os.path.join(root, "val")
            annotations_file = os.path.join(val_dir, "val_annotations.txt")
            with open(annotations_file, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    img_name, wnid = parts[0], parts[1]
                    self.file_paths.append(os.path.join(val_dir, "images", img_name))
                    self.targets.append(self._wnid_to_idx[wnid])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        pil_image = Image.open(self.file_paths[idx]).convert("RGB")
        target_cls = self.targets[idx]
        torch_image = pil_to_tensor(pil_image)
        torch_image = torchvision.tv_tensors.Image(torch_image)
        return torch_image, target_cls


def download_tiny_imagenet(dataset_path: str):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    os.makedirs(dataset_path, exist_ok=True)
    archive_path = os.path.join(dataset_path, "tiny-imagenet-200.zip")
    if not os.path.exists(archive_path):
        print("Downloading Tiny ImageNet...")
        download_with_progress(url, archive_path)
    print("Extracting Tiny ImageNet...")
    extract_zip_with_progress(archive_path, dataset_path)
    print("Cleanup Archive ... ")
    os.remove(archive_path)


def check_tiny_imagenet_ds_exists(dataset_path: str) -> bool:
    full_ds_path = os.path.join(dataset_path, "tiny-imagenet-200")
    if not os.path.exists(full_ds_path):
        return False
    try:
        contents = os.listdir(full_ds_path)
        return "train" in contents and "val" in contents and "wnids.txt" in contents
    except OSError:
        return False


if __name__ == "__main__":
    ds_path = os.path.join("..", "data", "tiny-imagenet")
    ds_present = check_tiny_imagenet_ds_exists(ds_path)
    print("Before Download: ", ds_present)
    if not ds_present:
        download_tiny_imagenet(ds_path)
    ds_present = check_tiny_imagenet_ds_exists(ds_path)
    print("After Download: ", ds_present)

    train_tiny = TinyImageNetDataset(ds_path, is_train=True)
    test_tiny = TinyImageNetDataset(ds_path, is_train=False)

    print(len(train_tiny))
    print(len(test_tiny))
    print(train_tiny[0])
    print(test_tiny[0])
