import os
import random
import shutil
from pathlib import Path

# ====== CONFIG ======
# Folder where your original class folders are
# Example structure:
# data/normal, data/abnormal, data/mi
SOURCE_DIR = Path("Split_ECG")

# Output base folder (will create train/val/test under this)
OUTPUT_DIR = Path("data")

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# File extensions to consider as images
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# =====================


def is_image_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in IMAGE_EXTS


def split_and_copy_class(class_name: str):
    src_class_dir = SOURCE_DIR / class_name

    if not src_class_dir.is_dir():
        print(f"[WARN] Skipping '{class_name}', not a directory: {src_class_dir}")
        return

    # Collect all image files
    files = [f for f in os.listdir(src_class_dir) if is_image_file(f)]
    if not files:
        print(f"[WARN] No images found in {src_class_dir}, skipping.")
        return

    random.shuffle(files)

    n_total = len(files)
    n_train = int(TRAIN_RATIO * n_total)
    n_val = int(VAL_RATIO * n_total)
    n_test = n_total - n_train - n_val  # whatever remains

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    print(f"[INFO] Class '{class_name}': total={n_total}, "
          f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    # Helper to copy a list of files into split/class folder
    def copy_files(file_list, split_name):
        dest_dir = OUTPUT_DIR / split_name / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        for fname in file_list:
            src = src_class_dir / fname
            dst = dest_dir / fname
            shutil.copy2(src, dst)

    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")


def main():
    random.seed(42)  # for reproducibility

    class_dirs = [
        d for d in os.listdir(SOURCE_DIR)
        if (SOURCE_DIR / d).is_dir()
    ]

    print(f"[INFO] Found classes: {class_dirs}")

    for class_name in class_dirs:
        split_and_copy_class(class_name)

    print("\n[DONE] Dataset split into train/val/test under:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
