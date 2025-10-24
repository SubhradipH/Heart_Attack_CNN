import os
import shutil
import random

def split_data(src_dir, dest_dir, train_ratio=0.7, val_ratio=0.15):
    # Get all subfolders (your 3 class folders)
    classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    print("Classes found:", classes)

    for cls in classes:
        src_path = os.path.join(src_dir, cls)
        images = os.listdir(src_path)
        random.shuffle(images)

        total = len(images)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))

        subsets = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }

        for subset, files in subsets.items():
            dest_path = os.path.join(dest_dir, subset, cls)
            os.makedirs(dest_path, exist_ok=True)

            for img in files:
                src_img = os.path.join(src_path, img)
                dst_img = os.path.join(dest_path, img)
                shutil.copy(src_img, dst_img)

        print(f"{cls}: {total} images split into "
              f"{len(subsets['train'])} train, {len(subsets['val'])} val, {len(subsets['test'])} test")

# ✅ Use your correct folder names here
split_data('Augmented_data', 'data_split')
