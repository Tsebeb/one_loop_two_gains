import os.path
import tarfile
import zipfile

from tqdm import tqdm
import urllib.request


def extract_tar_with_progress(tar_path: str, target_dir: str):
    with tarfile.open(tar_path, "r:*") as tar:
        members = tar.getmembers()

        with tqdm(total=len(members), unit="file") as pbar:
            for m in members:
                tar.extract(m, path=target_dir)
                pbar.update(1)


def extract_zip_with_progress(zip_path: str, dst_dir: str, chunk_size=1024 * 1024):
    os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        total_size = sum(m.file_size for m in members)

        with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Extracting {os.path.basename(zip_path)}") as pbar:
            for member in members:
                target_path = os.path.join(dst_dir, member.filename)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                if member.is_dir():
                    continue

                with zf.open(member) as src, open(target_path, "wb") as dst:
                    while True:
                        chunk = src.read(chunk_size)
                        if not chunk:
                            break
                        dst.write(chunk)
                        pbar.update(len(chunk))


def download_with_progress(url: str, dst: str):
    with tqdm(unit="B", unit_scale=True, unit_divisor=1024) as pbar:
        def reporthook(blocks, block_size, total_size):
            if total_size > 0:
                pbar.total = total_size
            pbar.update(blocks * block_size - pbar.n)

        urllib.request.urlretrieve(url, dst, reporthook)
