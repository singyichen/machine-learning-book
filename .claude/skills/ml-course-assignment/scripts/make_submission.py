#!/usr/bin/env python3
"""打包課程作業的繳交 zip，並確保中文檔名帶 UTF-8 旗標。

macOS 內建的 `zip` 指令不會設定 general purpose bit 11（UTF-8 檔名旗標），
老師在 Windows 解壓時中文檔名就會變成亂碼；而 `zip -UN=UTF8` 在系統內建版本
並不支援。Python 的 zipfile 遇到非 ASCII 檔名會自動設定該旗標，因此統一走這裡。

用法：

    python make_submission.py <作業目錄> <zip 檔名> <要打包的檔案> [更多檔案...]

範例：

    python make_submission.py lecture/07/assignment \\
        515661055_陳欣怡_Assignment3.zip \\
        07_assignment3_eval.py digits_pipeline_lr.pkl \\
        515661055_陳欣怡_Assignment3_結果報告.pdf

打包後會列出每個項目的大小與 UTF-8 旗標狀態供確認。
"""

import argparse
import sys
import zipfile
from pathlib import Path


UTF8_FLAG = 0x800


def build(assignment_dir, zip_name, filenames):
    base = Path(assignment_dir)
    if not base.is_dir():
        raise SystemExit(f"找不到作業目錄：{base}")

    missing = [name for name in filenames if not (base / name).exists()]
    if missing:
        raise SystemExit("下列檔案不存在，請先產生：\n  " + "\n  ".join(missing))

    output = base / zip_name
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in filenames:
            info = zipfile.ZipInfo.from_file(base / name, name)
            info.flag_bits |= UTF8_FLAG
            info.compress_type = zipfile.ZIP_DEFLATED
            with open(base / name, "rb") as source, archive.open(info, "w") as target:
                target.write(source.read())

    return output


def report(output):
    size_mb = output.stat().st_size / 1024 / 1024
    print(f"{output}  ({size_mb:.2f} MB)")
    with zipfile.ZipFile(output) as archive:
        for info in archive.infolist():
            flag = "是" if info.flag_bits & UTF8_FLAG else "否（純 ASCII 檔名，不需要）"
            print(f"  {info.file_size:>9}  UTF-8 旗標：{flag}  {info.filename}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="打包課程作業繳交 zip（中文檔名帶 UTF-8 旗標）"
    )
    parser.add_argument("assignment_dir", help="作業目錄，例如 lecture/07/assignment")
    parser.add_argument("zip_name", help="輸出檔名，例如 515661055_陳欣怡_Assignment3.zip")
    parser.add_argument("files", nargs="+", help="要放進 zip 的檔案（相對於作業目錄）")
    args = parser.parse_args(argv)

    report(build(args.assignment_dir, args.zip_name, args.files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
