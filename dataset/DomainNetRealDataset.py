import os
import shutil

import torchvision.tv_tensors
import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2.functional import pil_to_tensor

from dataset.download_utils import download_with_progress, extract_zip_with_progress


class DomainNetRealDataset(Dataset):
    """
    This class is a pytorch dataset wrapper for the Real Domain of the DomainNet Dataset
    from https://ai.bu.edu/M3SDA/#dataset and assumes
    """
    def __init__(self, ds_path: str, is_train: bool = False, use_resized: bool = True):
        super().__init__()
        if use_resized:
            dataset_core_name = "domain_net_real_256"
        else:
            dataset_core_name = "domain_net_real"

        if is_train:
            file_list_name = "real_train.txt"
        else:
            file_list_name = "real_test.txt"

        self._list_path = os.path.join(ds_path, dataset_core_name, file_list_name)
        with open(self._list_path, "r") as f:
            lines = f.readlines()

        self.file_paths = []
        self.targets = []
        # split by space / " "
        for l in lines:
            line_parts = l.strip().split(" ")
            assert len(line_parts) == 2, "Expected only the filename and the target dirs ... "
            file_path = os.path.join(ds_path, dataset_core_name, line_parts[0])
            self.file_paths.append(file_path)
            assert os.path.exists(file_path), ""
            self.targets.append(int(line_parts[1]))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        pil_image = Image.open(self.file_paths[idx])
        target_cls = self.targets[idx]
        torch_image = pil_to_tensor(pil_image)
        torch_image = torchvision.tv_tensors.Image(torch_image)
        return torch_image, target_cls


def download_domain_net_dataset(dataset_path: str) -> bool:
    url = "https://csr.bu.edu/ftp/visda/2019/multi-source/real.zip"
    url_train_list = "https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_train.txt"
    url_test_list = "https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_test.txt"
    os.makedirs(dataset_path, exist_ok=True)

    extracted_path = os.path.join(dataset_path, "domain_net_real")
    if not os.path.exists(extracted_path):
        archive_path = os.path.join(dataset_path, "real.zip")
        if not os.path.exists(archive_path):
            print("Downloading DomainNet - Real Domain ...")
            download_with_progress(url, archive_path)

        print("Extracting DomainNet - Real Domain ...")
        extract_zip_with_progress(archive_path, extracted_path)

        print("Cleanup ZIP ... ")
        os.remove(archive_path)

    train_path = os.path.join(dataset_path, "domain_net_real", "real_train.txt")
    if not os.path.exists(train_path):
        download_with_progress(url_train_list, train_path)

    test_path = os.path.join(dataset_path, "domain_net_real", "real_test.txt")
    if not os.path.exists(test_path):
        download_with_progress(url_test_list, test_path)

    if not check_domainnet_compressed_exists(dataset_path):
        compressed_path = os.path.join(dataset_path, "domain_net_real_256")
        os.makedirs(compressed_path, exist_ok=True)

        def _copy_folder_paths(src_folder, dst_folder):
            os.makedirs(dst_folder, exist_ok=True)
            files = os.listdir(src_folder)
            copy_folders = []

            for f in files:
                if os.path.isdir(os.path.join(src_folder, f)):
                    copy_folders += _copy_folder_paths(os.path.join(src_folder, f), os.path.join(dst_folder, f))
                if os.path.isfile(os.path.join(src_folder, f)):
                    if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg"):
                        if os.path.exists(os.path.join(dst_folder, f)):
                            continue  # Target already exists
                        copy_folders.append((os.path.join(src_folder, f), os.path.join(dst_folder, f)))
                    else:
                        print(f"found file with unexpected extension {f}")
                        continue
            return copy_folders

        copy_img_paths = _copy_folder_paths(os.path.join(extracted_path, "real"), os.path.join(compressed_path, "real"))
        # convert all images ...
        for src_path, dst_path in tqdm.tqdm(copy_img_paths, unit="img", desc="Compress image to (256x256) and save ... "):
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                img = img.resize((256, 256), Image.BILINEAR)
                img.save(dst_path)

        shutil.copy(train_path, os.path.join(compressed_path, "real_train.txt"))
        shutil.copy(test_path, os.path.join(compressed_path, "real_test.txt"))


def check_domainnet_compressed_exists(dataset_path: str) -> bool:
    full_ds_path = os.path.join(dataset_path, "domain_net_real_256")
    if not os.path.exists(full_ds_path):
        return False
    else:
        try:
            current_files = os.listdir(full_ds_path)
            if "real_train.txt" in current_files and "real_test.txt" in current_files and "real" in current_files:
                return True
            else:
                return False
        except:
            return False


if __name__ == "__main__":
    ds_path = os.path.join("..", "data", "domain_net")
    ds_present = check_domainnet_compressed_exists(ds_path)
    print(ds_present)
    if not ds_present:
        download_domain_net_dataset(ds_path)
    ds_present = check_domainnet_compressed_exists(ds_path)
    print(ds_present)
