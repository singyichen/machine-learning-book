"""讓 torchvision 接受本機重建的 CelebA（不繳交）。

torchvision.datasets.CelebA 會對五個標註檔逐一比對硬編碼的 md5，而官方檔案
只在 Google Drive 上（長期配額封鎖）。prepare_celeba_local.py 由 HuggingFace
鏡像重建的標註檔內容正確、行尾格式不同，md5 因此對不上，資料集會被判定為
corrupted。

本模組把 file_list 裡的期望 md5 換成本機檔案的實際 md5，只影響「完整性檢查」
這一步，不改變任何資料讀取行為。

繳交的 model_1_test.py / model_2_test.py 不會 import 本模組 —— 助教環境裡的
CelebA 是官方原檔，md5 本來就吻合。本機要驗證繳交腳本時，用：

    python -c "import celeba_local_patch, runpy; \\
               celeba_local_patch.apply(); \\
               runpy.run_path('model_1_test.py', run_name='__main__')"
"""

from pathlib import Path
import hashlib

CELEBA_ROOT = Path(__file__).resolve().parent.parent / "CelebA"


def _md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply(root=None):
    """把 CelebA.file_list 的期望 md5 改成本機檔案的實際值。回傳改過的檔名。"""
    from torchvision.datasets import CelebA

    base = Path(root) if root is not None else CELEBA_ROOT
    base = base / "celeba"

    patched = []
    new_list = []
    for file_id, md5, filename in CelebA.file_list:
        path = base / filename
        if path.exists():
            md5 = _md5(path)
            patched.append(filename)
        new_list.append((file_id, md5, filename))
    CelebA.file_list = new_list
    return patched
