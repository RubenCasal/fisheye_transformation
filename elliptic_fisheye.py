import cv2
import numpy as np

def circular_fisheye(img: np.ndarray, strength=0.7) -> np.ndarray:
    """
    Applies a **perfectly circular** fisheye transformation by scaling
    based on the smallest image dimension to ensure a **true** circular projection.

    Args:
        img (numpy.ndarray): Input image (any aspect ratio).
        strength (float): Fisheye distortion strength (0.5–1.0 typical).

    Returns:
        np.ndarray: A circular fisheye image.
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # Use the smallest dimension for a **true circular mapping**
    R_max = min(w, h) / 2  

    # Define square output canvas
    out_size = int(2 * R_max)
    fisheye_img = np.zeros((out_size, out_size, 3), dtype=img.dtype)

    # New center in the output
    new_center = (out_size // 2, out_size // 2)

    for y in range(out_size):
        for x in range(out_size):
            # Convert to local coordinates (normalized to be **perfectly circular**)
            dx = (x - new_center[0]) / R_max  
            dy = (y - new_center[1]) / R_max  
            r = np.sqrt(dx**2 + dy**2)

            # Ignore points outside the circular boundary
            if r > 1.0:
                continue

            # Compute angle
            theta = np.arctan2(dy, dx)

            # Nonlinear fisheye radial transformation
            r_dist = np.tan(r * (np.pi / 2) * strength) * R_max / np.tan((np.pi / 2) * strength)

            # Map back to the original image coordinates **ensuring a circular projection**
            src_x = int(center[0] + r_dist * np.cos(theta))
            src_y = int(center[1] + r_dist * np.sin(theta))

            # Assign valid pixels
            if 0 <= src_x < w and 0 <= src_y < h:
                fisheye_img[y, x] = img[src_y, src_x]

    return fisheye_img


if __name__ == "__main__":
    # Load input image
    input_path = "image.png"
    output_path = "output_fisheye_circular.webp"
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Could not load: {input_path}")

    # Apply perfectly circular fisheye transformation
    fisheye_result = circular_fisheye(image, strength=0.7)

    # Save & show
    cv2.imwrite(output_path, fisheye_result)
    cv2.imshow("Perfect Circular Fisheye", fisheye_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
