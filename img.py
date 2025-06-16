from PIL import Image
import os

def crop_ecg_img(input_folder, output_folder="normal_leads"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # DEBUG: print resolved folder and files inside
    print("Resolved input folder:", os.path.abspath(input_folder))
    try:
        files_in_folder = os.listdir(input_folder)
        print("Files found:", files_in_folder)
    except FileNotFoundError:
        print("ERROR: The specified input folder does not exist.")
        return

    image_files = [f for f in files_in_folder if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    if not image_files:
        print("No image files found in the input folder.")
        return

    box_coordinates = [
        (150, 340, 640, 540), (645, 340, 1135, 540), (1140, 340, 1630, 540), (1650, 340, 2140, 540),
        (150, 640, 640, 840), (645, 640, 1135, 848), (1140, 640, 1630, 840), (1650, 640, 2140, 840),
        (150, 950, 640, 1150), (645, 950, 1135, 1150), (1150, 950, 1640, 1150), (1650, 950, 2140, 1150)
    ]

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        try:
            img = Image.open(image_path)
        except Exception as e:
            print(f"Failed to open {image_file}: {e}")
            continue

        img = img.resize((2213, 1572))
        base_name = os.path.splitext(image_file)[0]

        for i, cords in enumerate(box_coordinates, 1):
            box = img.crop(cords)
            output_path = os.path.join(output_folder, f'{base_name}_lead_{i}.jpg')
            box.convert('RGB').save(output_path, 'JPEG', quality=95)
            print(f'Saved lead {i} from {image_file} to {output_path}')
    
    print(f"Successfully processed {len(image_files)} images and extracted {len(box_coordinates)} leads per image.")

if __name__ == "__main__":
    input_folder_path = r"D:\CNN_H.A\ECG\Normal Person ECG Images (284x12=3408)"
    crop_ecg_img(input_folder_path)
