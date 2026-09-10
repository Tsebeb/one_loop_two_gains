from torchvision.datasets import ImageFolder
import os
from dataset.download_utils import download_with_progress, extract_tar_with_progress


class ImageWoofDataset(ImageFolder):
    """
    This class is a pytorch dataset wrapper for the ImageWoof dataset.
    The dataset is a subset of 10 dog breeds from the ImageNet dataset
    The dataset is available at https://github.com/fastai/imagenette
    """
    def __init__(self, ds_path: str, is_train: bool = False):
        ds_path = os.path.join(ds_path, "imagewoof2-320", "train" if is_train else "val")
        super().__init__(ds_path)


def download_imagewoof_ds(dataset_path: str) -> bool:
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz"
    os.makedirs(dataset_path, exist_ok=True)
    archive_path = os.path.join(dataset_path, "imagewoof2-320.tgz")
    if not os.path.exists(archive_path):
        print("Downloading ImageWoof...")
        download_with_progress(url, archive_path)
    print("Extracting ImageWoof...")
    extract_tar_with_progress(archive_path, dataset_path)
    print("Cleanup Archive ... ")
    os.remove(archive_path)


def check_imagewoof_ds_exists(dataset_path: str) -> bool:
    full_ds_path = os.path.join(dataset_path, "imagewoof2-320")
    if not os.path.exists(full_ds_path):
        return False
    else:
        try:
            current_files = os.listdir(full_ds_path)
            if "noisy_imagewoof.csv" in current_files:
                return True
            else:
                return False
        except:
            return False


if __name__ == "__main__":
    ds_path = os.path.join("..", "data", "imagewoof")
    ds_present = check_imagewoof_ds_exists(ds_path)
    print(ds_present)
    if not ds_present:
        download_imagewoof_ds(ds_path)
    ds_present = check_imagewoof_ds_exists(ds_path)
    print(ds_present)
