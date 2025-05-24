import os
import cv2
import numpy as np

# Define the folder containing the mask images
folder_path = "../mask2former/solarPV/Multi_Model/test_gt"  # Change this to your actual folder path

# Define the mapping
value_map = {0: 255, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}



def process_masks(folder_path):
    """
    Processes mask images in the folder by applying pixel value transformations
    based on the predefined mapping.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith((".png", ".jpg", ".tif")):  # Process image files only
            file_path = os.path.join(folder_path, filename)

            # Read the image in grayscale mode (as a mask)
            mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Skipping {filename} - Not a valid image")
                continue
            
            # Apply the transformation using numpy
            transformed_mask = np.vectorize(lambda x: value_map.get(x, x))(mask).astype(np.uint8)

            # Save the transformed image (overwrite the original)
            cv2.imwrite(file_path, transformed_mask)
            print(f"Processed and saved: {filename}")

    print("All masks processed successfully.")

def analyze_mask_values(folder_path):
    """
    Goes through all mask images in the folder and prints:
    - The filename of the mask being analyzed
    - The unique pixel values present in the mask
    """
    for filename in os.listdir(folder_path):
        if filename.endswith((".png", ".jpg", ".tif")):  # Process image files only
            file_path = os.path.join(folder_path, filename)

            # Read the image in grayscale mode
            mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Skipping {filename} - Not a valid image")
                continue
            
            # Get unique pixel values in the mask
            unique_values = np.unique(mask)
            
            # Print results
            print(f"Mask: {filename}")
            print(f"Unique Values: {unique_values}")
            print("-" * 40)

# Example usage
process_masks(folder_path)  # Apply transformations/
# analyze_mask_values(folder_path)  # Analyze masks/