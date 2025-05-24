import os
from PIL import Image
import shutil

# === Settings ===
img_input_folder = '/home/esteban/SODS-Former/mask2former/solarPV/Multi_Model/train_imgs'
mask_input_folder = '/home/esteban/SODS-Former/mask2former/solarPV/Multi_Model/train_gt'
output_folder = '/home/esteban/SODS-Former/mask2former/solarPV/DifSize'  # Output root directory
threshold = 40                   # Allowed dimension variation

# === Helper Function ===
def is_similar_size(size1, size2, threshold):
    return abs(size1[0] - size2[0]) <= threshold and abs(size1[1] - size2[1]) <= threshold

# === Main Logic ===
def group_images_with_masks(img_input_folder, mask_input_folder, output_folder, threshold):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    groups = []        # Representative sizes
    group_paths = []   # Corresponding folder paths

    for filename in os.listdir(img_input_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            continue  # Skip non-image files

        img_path = os.path.join(img_input_folder, filename)
        with Image.open(img_path) as img:
            size = img.size  # (width, height)

        placed = False
        for i, rep_size in enumerate(groups):
            if is_similar_size(size, rep_size, threshold):
                target_folder = group_paths[i]
                placed = True
                break

        if not placed:
            target_folder = os.path.join(output_folder, f"{size[0]}x{size[1]}")
            os.makedirs(os.path.join(target_folder, 'test_imgs'), exist_ok=True)
            os.makedirs(os.path.join(target_folder, 'test_gt'), exist_ok=True)
            groups.append(size)
            group_paths.append(target_folder)

        # Copy image
        dest_img_path = os.path.join(target_folder, 'test_imgs', filename)
        shutil.copy(img_path, dest_img_path)

        # Handle mask
        base_name, _ = os.path.splitext(filename)
        mask_filename = base_name + '_Mask.png'
        mask_path = os.path.join(mask_input_folder, mask_filename)

        if os.path.exists(mask_path):
            dest_mask_path = os.path.join(target_folder, 'test_gt', mask_filename)
            shutil.copy(mask_path, dest_mask_path)
        else:
            print(f"⚠️ Mask not found for: {filename}")

# === Run It ===
group_images_with_masks(img_input_folder, mask_input_folder, output_folder, threshold)