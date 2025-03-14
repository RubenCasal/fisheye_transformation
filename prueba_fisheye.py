import cv2
import numpy as np

def perfect_circle_fisheye(img: np.ndarray, strength=0.7) -> np.ndarray:
    """
    Applies a fisheye transformation that forces a perfectly circular output.
    The output image will be a square of side 2*R_max, so the circle is centered.
    
    Args:
        img (numpy.ndarray): The input image (any aspect ratio).
        strength (float): Controls the non-linear distortion (0.5–1.0 for typical ranges).
    
    Returns:
        np.ndarray: A square image containing a perfect circle fisheye view.
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    R_max = min(center)  # Maximum radius that fits in the original image

    # Define output size: a square with side = 2*R_max
    out_size = 2 * R_max
    fisheye_img = np.zeros((out_size, out_size, 3), dtype=img.dtype)

    # New center in the output image
    new_center = (out_size // 2, out_size // 2)

    # Loop over every pixel in the output (square)
    for y in range(out_size):
        for x in range(out_size):
            # Convert (x, y) to local coords
            dx = x - new_center[0]
            dy = y - new_center[1]
            r = np.sqrt(dx**2 + dy**2)

            # Skip anything outside our circle
            if r > R_max:
                continue

            # Compute angle
            theta = np.arctan2(dy, dx)

            # Nonlinear fisheye formula:
            #    r_distorted = tan( (r / R_max)*(π/2)*strength ) * (R_max / tan((π/2)*strength))
            # Explanation:
            #  (r / R_max) normalizes radius [0..1],
            #  multiplied by (π/2)*strength for the distortion,
            #  use tan(...) to amplify edges,
            #  scaled back by dividing out tan(...) at R_max.
            r_dist = np.tan( (r / R_max) * (np.pi / 2) * strength ) \
                      * R_max / np.tan( (np.pi / 2) * strength )

            # Map back to original image coordinates
            src_x = int(center[0] + r_dist * np.cos(theta))
            src_y = int(center[1] + r_dist * np.sin(theta))

            # Assign pixel if valid
            if 0 <= src_x < w and 0 <= src_y < h:
                fisheye_img[y, x] = img[src_y, src_x]

    return fisheye_img


# Example usage
if __name__ == "__main__":
    input_path = "image.png"    # Your input image
    output_path = "output_fisheye_perfect_circle.webp"
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Could not load: {input_path}")

    # Apply perfect circle fisheye
    fisheye_result = perfect_circle_fisheye(image, strength=0.7)

    # Save & show
    cv2.imwrite(output_path, fisheye_result)
    cv2.imshow("Perfect Circle Fisheye", fisheye_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
