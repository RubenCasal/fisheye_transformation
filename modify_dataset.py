import cv2
import numpy as np
import os
from pathlib import Path
import glob

# === CONFIG === #
DATASET_DIR = "roboflow_dataset"  # Change this to your dataset path
OUTPUT_DIR = "roboflow_fisheye"   # Output directory
STRENGTH = 0.7                    # Distortion strength

# Ensure output directories exist
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels", exist_ok=True)

def fisheye_transform(img: np.ndarray, strength=0.7):
    """
    Applies fisheye transformation with circular output.
    Args:
        img (np.ndarray): Input image.
        strength (float): Distortion strength (0.5–1.0 recommended).
    Returns:
        np.ndarray: Transformed image.
        function: Mapping function to transform bounding boxes.
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    R_max = min(center)

    # Define output size: a square with side 2*R_max
    out_size = 2 * R_max
    fisheye_img = np.zeros((out_size, out_size, 3), dtype=img.dtype)

    # New center in output image
    new_center = (out_size // 2, out_size // 2)

    def transform_bbox(x, y):
        """ Apply fisheye transformation to bounding box coordinates. """
        dx = x - center[0]
        dy = y - center[1]
        r = np.sqrt(dx**2 + dy**2)

        if r > R_max:
            return None  # Ignore points outside the transformed area

        theta = np.arctan2(dy, dx)
        r_dist = np.tan((r / R_max) * (np.pi / 2) * strength) * R_max / np.tan((np.pi / 2) * strength)

        new_x = int(new_center[0] + r_dist * np.cos(theta))
        new_y = int(new_center[1] + r_dist * np.sin(theta))

        return new_x, new_y

    for y in range(out_size):
        for x in range(out_size):
            dx = x - new_center[0]
            dy = y - new_center[1]
            r = np.sqrt(dx**2 + dy**2)

            if r > R_max:
                continue

            theta = np.arctan2(dy, dx)
            r_dist = np.tan((r / R_max) * (np.pi / 2) * strength) * R_max / np.tan((np.pi / 2) * strength)

            src_x = int(center[0] + r_dist * np.cos(theta))
            src_y = int(center[1] + r_dist * np.sin(theta))

            if 0 <= src_x < w and 0 <= src_y < h:
                fisheye_img[y, x] = img[src_y, src_x]

    return fisheye_img, transform_bbox


def process_dataset(dataset_dir, output_dir, strength=0.7):
    """
    Process the YOLO dataset from Roboflow, apply fisheye transformation, 
    and modify the labels accordingly.
    """
    image_paths = glob.glob(os.path.join(dataset_dir, "images", "*.jpg"))
    image_paths += glob.glob(os.path.join(dataset_dir, "images", "*.png"))
    label_paths = {Path(img).stem: os.path.join(dataset_dir, "labels", Path(img).stem + ".txt") for img in image_paths}

    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Skipping {image_path}, cannot load image.")
            continue

        fisheye_img, transform_bbox = fisheye_transform(img, strength)

        # Save new image
        output_img_path = os.path.join(output_dir, "images", Path(image_path).name)
        cv2.imwrite(output_img_path, fisheye_img)

        # Process corresponding label
        label_path = label_paths.get(Path(image_path).stem)
        if not label_path or not os.path.exists(label_path):
            continue  # Skip if no corresponding label file

        with open(label_path, "r") as f:
            lines = f.readlines()

        new_labels = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id, x_center, y_center, width, height = map(float, parts)

            # Convert normalized coordinates to pixel values
            x_center *= img.shape[1]
            y_center *= img.shape[0]
            width *= img.shape[1]
            height *= img.shape[0]

            # Apply fisheye transformation to bbox center
            new_center = transform_bbox(x_center, y_center)
            if new_center is None:
                continue  # Ignore bounding
