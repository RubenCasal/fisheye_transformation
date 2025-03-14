import cv2
import numpy as np

def rectangular_to_circular(image):
    """
    Rearranges a rectangular image into a circular format using radial coordinate mapping.

    Args:
        image (np.ndarray): Input rectangular image.

    Returns:
        np.ndarray: Circularly remapped image.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    max_radius = min(center)  # Define the largest circle that fits

    # Create a blank canvas for the circular image
    circular_img = np.zeros((h, w, 3), dtype=np.uint8)

    # Create a polar mapping
    for y in range(h):
        for x in range(w):
            # Convert Cartesian to Polar Coordinates
            dx = x - center[0]
            dy = y - center[1]
            r = np.hypot(dx, dy)  # Radius
            theta = np.arctan2(dy, dx)  # Angle

            # Map back to image coordinates
            src_x = int((theta + np.pi) / (2 * np.pi) * w)
            src_y = int(r / max_radius * h / 2)

            # Assign pixel if within bounds
            if 0 <= src_x < w and 0 <= src_y < h:
                circular_img[y, x] = image[src_y, src_x]

    return circular_img

# Load the input image
input_image = cv2.imread("img.webp")

# Convert to circular format
circular_image = rectangular_to_circular(input_image)

# Save and display
cv2.imwrite("circular_image.jpg", circular_image)
cv2.imshow("Circular Projection", circular_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
