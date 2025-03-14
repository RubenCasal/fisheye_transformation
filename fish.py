import cv2
import numpy as np
from math import sqrt

# Set the image file path
IMAGE_PATH = "img.webp"  # Change this to your image file name
OUTPUT_PATH = "fisheye_output.png"

# Distortion strength (Higher values = stronger fisheye effect)
DISTORTION_COEFFICIENT = 0.5  

def get_fish_xn_yn(source_x, source_y, radius, distortion):
    """
    Compute the new pixel coordinates after applying fisheye distortion.
    """
    factor = 1 - distortion * (radius**2)
    if factor == 0:
        return source_x, source_y
    return source_x / factor, source_y / factor

def apply_fisheye_effect(image, distortion_coefficient):
    """
    Apply a fisheye effect to the input image.
    """
    height, width = image.shape[:2]
    output_image = np.zeros_like(image)

    for x in range(width):
        for y in range(height):
            # Normalize x and y between [-1, 1]
            xnd, ynd = (2*x - width)/width, (2*y - height)/height
            radius = sqrt(xnd**2 + ynd**2)

            # Get distorted coordinates
            x_new, y_new = get_fish_xn_yn(xnd, ynd, radius, distortion_coefficient)

            # Convert back to image coordinates
            x_new = int(((x_new + 1) * width) / 2)
            y_new = int(((y_new + 1) * height) / 2)

            # Assign new pixel values if within image bounds
            if 0 <= x_new < width and 0 <= y_new < height:
                output_image[y, x] = image[y_new, x_new]

    return output_image

# Load image
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise ValueError(f"Error loading image '{IMAGE_PATH}'. Check the file path.")

# Apply fisheye effect
fisheye_image = apply_fisheye_effect(image, DISTORTION_COEFFICIENT)

# Save and display the output
cv2.imwrite(OUTPUT_PATH, fisheye_image)
