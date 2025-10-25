import os
from PIL import Image, ImageOps

# Folder containing your original images
input_folder = r"D:\CNN_H.A\Split_ECG\Normal Person ECG Images (284x12=3408)"

# Output folder
output_folder = r"D:\CNN_H.A\Augmented_data\Normal Person ECG Images (284x12=3408)_AUG"
os.makedirs(output_folder, exist_ok=True)

# Counter for naming images
counter = 1

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # process only images
        input_path = os.path.join(input_folder, filename)
        try:
            img = Image.open(input_path).convert("RGB")
        except Exception as e:
            print(f"Failed to open {filename}: {e}")
            continue

        # Augmentations
        augmented = {
            "original": img,
            "rotated_90": img.rotate(90, expand=True),
            "flipped_horizontal": ImageOps.mirror(img),
            "rotated_180": img.rotate(180, expand=True)
        }

        # Save all images
        for aug_name, im in augmented.items():
            save_name = f"NORMAL_{counter}.jpg"
            save_path = os.path.join(output_folder, save_name)
            im.save(save_path, format="JPEG", quality=95)
            counter += 1

print(f"Finished! All augmented images are saved in {output_folder}")
import os
from PIL import Image, ImageOps

# Folder containing your original images
input_folder = r"D:\CNN_H.A\Split_ECG\Normal Person ECG Images (284x12=3408)"

# Output folder
output_folder = r"D:\CNN_H.A\Augmented_data\Normal Person ECG Images (284x12=3408)_AUG"
os.makedirs(output_folder, exist_ok=True)

# Counter for naming images
counter = 1

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # process only images
        input_path = os.path.join(input_folder, filename)
        try:
            img = Image.open(input_path).convert("RGB")
        except Exception as e:
            print(f"Failed to open {filename}: {e}")
            continue

        # Augmentations
        augmented = {
            "original": img,
            "rotated_90": img.rotate(90, expand=True),
            "flipped_horizontal": ImageOps.mirror(img),
            "rotated_180": img.rotate(180, expand=True)
        }

        # Save all images
        for aug_name, im in augmented.items():
            save_name = f"NORMAL_{counter}.jpg"
            save_path = os.path.join(output_folder, save_name)
            im.save(save_path, format="JPEG", quality=95)
            counter += 1

print(f"Finished! All augmented images are saved in {output_folder}")

