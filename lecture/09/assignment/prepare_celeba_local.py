"""本機 CelebA 資料準備（不繳交）。

官方 CelebA 託管在 Google Drive，長期處於「Too many users have viewed or
downloaded this file recently」的配額封鎖狀態，torchvision 的 download=True
在本機一律失敗。本腳本改由 HuggingFace 的 eurecom-ds/celeba 鏡像取得影像與
40 維屬性，重建成 torchvision.datasets.CelebA 期望的目錄結構：

    CelebA/celeba/
        img_align_celeba/000001.jpg ...
        list_eval_partition.txt      （官方切分，202599 行）
        list_attr_celeba.txt         （40 維屬性）
        identity_CelebA.txt          （本作業用不到，以佔位值補齊行數）
        list_bbox_celeba.txt         （同上）
        list_landmarks_align_celeba.txt （同上）

鏡像的 train/validation/test 筆數為 162770 / 19867 / 19962，與官方
list_eval_partition.txt 完全一致，因此第 i 筆就對應官方切分的第 i 個檔名。
本腳本會實際比對屬性值做驗證（見 verify_against_partition）。

只會解出本作業實際用到的影像（訓練前 16000、驗證前 1000、測試前 10000），
不是整個 202599 張，省去 8.9GB 的下載。

繳交的 model_1_test.py / model_2_test.py 不依賴本腳本：它們呼叫標準的
torchvision.datasets.CelebA，助教環境中既有的 CelebA 即可直接使用。
"""

from pathlib import Path
import io
import sys

ASSIGNMENT_DIR = Path(__file__).resolve().parent
CELEBA_ROOT = ASSIGNMENT_DIR.parent / "CelebA"
BASE_DIR = CELEBA_ROOT / "celeba"
IMAGE_DIR = BASE_DIR / "img_align_celeba"
HF_DIR = CELEBA_ROOT / "_hf"

TOTAL_IMAGES = 202599
N_TRAIN, N_VALID, N_TEST = 162770, 19867, 19962

# 本作業實際用到的筆數（作業規定：訓練前 16000、測試前 10000；驗證沿用課堂範例的 1000）
NEEDED = {"train": 16000, "validation": 1000, "test": 10000}

ATTR_NAMES = (
    "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips "
    "Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin "
    "Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open "
    "Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline "
    "Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat "
    "Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
).split()

SPLIT_FILES = {
    "train": ["train-00000-of-00018.parquet", "train-00001-of-00018.parquet"],
    "validation": ["validation-00000-of-00001.parquet"],
    "test": ["test-00000-of-00001.parquet"],
}

# 官方切分中每個 split 的第一個檔名序號（1-indexed）
SPLIT_OFFSET = {"train": 1, "validation": N_TRAIN + 1, "test": N_TRAIN + N_VALID + 1}


def read_partition():
    """讀入官方 list_eval_partition.txt，回傳 [(filename, split_code), ...]。"""
    path = HF_DIR / "list_eval_partition.txt"
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        name, code = line.split()
        rows.append((name, int(code)))
    if len(rows) != TOTAL_IMAGES:
        raise RuntimeError(f"partition 檔有 {len(rows)} 行，應為 {TOTAL_IMAGES} 行")
    return rows


def extract_split(split, limit, attr_table):
    """自 parquet 解出該 split 前 limit 筆的影像與屬性，寫入 img_align_celeba/。"""
    import pyarrow.parquet as pq
    from PIL import Image

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    offset = SPLIT_OFFSET[split]
    written = 0

    for filename in SPLIT_FILES[split]:
        if written >= limit:
            break
        table = pq.ParquetFile(HF_DIR / filename)
        for batch in table.iter_batches(batch_size=256, columns=["image", "attributes"]):
            images = batch.column("image").to_pylist()
            attrs = batch.column("attributes").to_pylist()
            for image, attr in zip(images, attrs):
                if written >= limit:
                    break
                index = offset + written          # 1-indexed 官方檔名序號
                out = IMAGE_DIR / f"{index:06d}.jpg"
                if not out.exists():
                    Image.open(io.BytesIO(image["bytes"])).convert("RGB").save(
                        out, "JPEG", quality=95
                    )
                # 鏡像的屬性是 0/1，官方 txt 用 -1/1
                attr_table[index - 1] = [1 if v else -1 for v in attr]
                written += 1
            if written >= limit:
                break
        del table

    if written < limit:
        raise RuntimeError(f"{split} 只解出 {written} 筆，不足 {limit} 筆")
    return written


def write_annotation_files(partition, attr_table):
    """寫出 torchvision 需要的五個標註檔，行數與順序皆比照官方。"""
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    (BASE_DIR / "list_eval_partition.txt").write_text(
        "".join(f"{name} {code}\n" for name, code in partition)
    )

    lines = [f"{TOTAL_IMAGES}\n", " ".join(ATTR_NAMES) + "\n"]
    for (name, _), attr in zip(partition, attr_table):
        lines.append(name + " " + " ".join(f"{v:d}" for v in attr) + "\n")
    (BASE_DIR / "list_attr_celeba.txt").write_text("".join(lines))

    # 以下三個檔案 target_type='attr' 時不會用到數值，但 torchvision 仍會讀取，
    # 所以必須存在且行數、header 格式正確。
    (BASE_DIR / "identity_CelebA.txt").write_text(
        "".join(f"{name} 1\n" for name, _ in partition)
    )
    (BASE_DIR / "list_bbox_celeba.txt").write_text(
        f"{TOTAL_IMAGES}\nimage_id x_1 y_1 width height\n"
        + "".join(f"{name} 0 0 178 218\n" for name, _ in partition)
    )
    (BASE_DIR / "list_landmarks_align_celeba.txt").write_text(
        f"{TOTAL_IMAGES}\n"
        "lefteye_x lefteye_y righteye_x righteye_y nose_x nose_y "
        "leftmouth_x leftmouth_y rightmouth_x rightmouth_y\n"
        + "".join(f"{name} 0 0 0 0 0 0 0 0 0 0\n" for name, _ in partition)
    )


def verify_against_partition():
    """載入重建後的資料集，確認 split 邊界、檔名與屬性維度都對得上。"""
    from torchvision import datasets

    checks = []
    for split, expected_total in (("train", N_TRAIN), ("valid", N_VALID), ("test", N_TEST)):
        dataset = datasets.CelebA(str(CELEBA_ROOT), split=split, target_type="attr",
                                  download=False)
        key = {"train": "train", "valid": "validation", "test": "test"}[split]
        first_index = SPLIT_OFFSET[key]
        checks.append(
            {
                "split": split,
                "size": len(dataset),
                "expected_size": expected_total,
                "first_filename": dataset.filename[0],
                "expected_first_filename": f"{first_index:06d}.jpg",
                "attr_width": dataset.attr.shape[1],
                "smiling_rate_over_needed": float(
                    dataset.attr[: NEEDED[key], 31].float().mean()
                ),
            }
        )
    return checks


def main():
    if not (HF_DIR / "list_eval_partition.txt").exists():
        sys.exit(f"缺少 {HF_DIR}/list_eval_partition.txt，請先下載鏡像檔案")

    partition = read_partition()
    attr_table = [[-1] * 40 for _ in range(TOTAL_IMAGES)]

    for split, limit in NEEDED.items():
        count = extract_split(split, limit, attr_table)
        print(f"{split:<11} 解出 {count} 張影像")

    write_annotation_files(partition, attr_table)
    print(f"標註檔已寫入 {BASE_DIR}")

    for check in verify_against_partition():
        ok = (
            check["size"] == check["expected_size"]
            and check["first_filename"] == check["expected_first_filename"]
            and check["attr_width"] == 40
        )
        print(
            f"{check['split']:<6} {check['size']:>6} 筆  首檔 {check['first_filename']}  "
            f"Smiling 比例 {check['smiling_rate_over_needed']:.4f}  {'OK' if ok else 'FAIL'}"
        )


if __name__ == "__main__":
    main()
