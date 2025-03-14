import cv2
import numpy as np


def make_square(image: np.ndarray) -> np.ndarray:
    """
    Resize the image into a square by padding with black pixels.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Square image with black padding.
    """
    h, w = image.shape[:2]
    square_size = max(h, w)  # New size (max of width and height)

    # Create a black background
    square_img = np.zeros((square_size, square_size, 3), dtype=np.uint8)

    # Compute offsets to center the image
    x_offset = (square_size - w) // 2
    y_offset = (square_size - h) // 2

    # Place the original image in the center
    square_img[y_offset:y_offset+h, x_offset:x_offset+w] = image
    return square_img
def fisheye_to_circular(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print("Error: Could not load the image!")
        return
    image = make_square(image)
    # Get image dimensions
    h, w = image.shape[:2]
    center = (w // 2, h // 2)  # Image center
    max_radius = min(center)   # Radius for circular mask

    # Apply polar transformation to remap pixels into circular format
    circular_img = cv2.linearPolar(image, center, max_radius, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)

    # Create circular mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, max_radius, 255, -1)  # White filled circle

    # Apply the mask
    circular_img = cv2.bitwise_and(circular_img, circular_img, mask=mask)

    # Save the result
    cv2.imwrite(output_path, circular_img)
    print(f"Circular fisheye image saved as: {output_path}")

# Example usage
fisheye_to_circular("./output/img_fisheye.webp", "output_circular.png")
