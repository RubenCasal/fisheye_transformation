import cv2
import numpy as np
import scipy.interpolate
import logging
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

### === HARDCODED CAMERA PARAMETERS === ###
# Intrinsic Camera Matrix (K)
CAM_INTR = np.array([
    [284.509100, 0.000000, 421.896335],
    [0.000000, 282.941856, 398.100316],
    [0.000000, 0.000000, 1.000000]
], dtype=np.float32)

# Distortion Coefficients (D) -> (k1, k2, p1, p2)
DIST_COEFF = np.array([-0.014216, 0.060412, -0.054711, 0.011151], dtype=np.float32)

# Output directory
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def interpolate_to_circular(image: np.ndarray) -> np.ndarray:
    """
    Transform the original rectangular image into a circular format using interpolation.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Circularly interpolated image.
    """
    h, w = image.shape[:2]
    radius = min(h, w) // 2  # Use the smaller dimension to define the circular radius
    center = (w // 2, h // 2)

    # Create a circular grid
    ys, xs = np.indices((h, w))
    xs = xs - center[0]
    ys = ys - center[1]
    r = np.sqrt(xs**2 + ys**2)
    theta = np.arctan2(ys, xs)  # Convert to polar coordinates

    # Normalize radius to fit within the circular boundary
    r = (r / np.max(r)) * radius

    # Convert polar coordinates back to cartesian
    x_new = center[0] + r * np.cos(theta)
    y_new = center[1] + r * np.sin(theta)

    # Interpolate values from the original image
    interpolators = [
        scipy.interpolate.RegularGridInterpolator((np.arange(h), np.arange(w)), image[:, :, channel], method="linear",
                                                 bounds_error=False, fill_value=0)
        for channel in range(image.shape[2])
    ]

    # Apply interpolation for each channel
    interpolated_img = np.dstack([interpolator((y_new, x_new)) for interpolator in interpolators])

    # Convert to valid image type
    interpolated_img = np.clip(np.round(interpolated_img), 0, 255).astype(np.uint8)

    return interpolated_img


def distort_image(img: np.ndarray) -> np.ndarray:
    """
    Apply fisheye distortion using hardcoded camera parameters.

    Args:
        img (numpy.ndarray): Input image in BGR format.

    Returns:
        numpy.ndarray: Distorted output image.
    """
    h, w = img.shape[:2]

    # Generate pixel coordinate grid
    xs = np.arange(w)
    ys = np.arange(h)
    xv, yv = np.meshgrid(xs, ys)
    img_pts = np.stack((xv, yv), axis=2).reshape((-1, 1, 2)).astype(np.float32)

    # Compute mapping from undistorted to distorted space
    undistorted_px = cv2.fisheye.undistortPoints(img_pts, CAM_INTR, DIST_COEFF)
    undistorted_px = cv2.convertPointsToHomogeneous(undistorted_px)
    undistorted_px = np.tensordot(undistorted_px, CAM_INTR, axes=(2, 1))
    undistorted_px = cv2.convertPointsFromHomogeneous(undistorted_px).reshape((h, w, 2))

    # Flip coordinates (OpenCV uses height-first format)
    undistorted_px = np.flip(undistorted_px, axis=2)

    # Interpolation (Linear for RGB images)
    interpolators = [scipy.interpolate.RegularGridInterpolator(
        (ys, xs), img[:, :, channel], method="linear", bounds_error=False, fill_value=0
    ) for channel in range(img.shape[2])]

    # Apply interpolation to obtain distorted image
    img_dist = np.dstack([interpolator(undistorted_px) for interpolator in interpolators])

    # Convert to valid image type
    img_dist = np.clip(np.round(img_dist), 0, 255).astype(np.uint8)

    return img_dist


def process_image(input_path: Path):
    """
    Process a single image:
      - Convert to circular format using interpolation.
      - Apply fisheye distortion and save.

    Args:
        input_path (Path): Path to the input image.
    """
    log.info(f"Processing: {input_path}")

    # Read input image
    img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img is None:
        log.error(f"Could not load image: {input_path}")
        return

    # Convert image to circular format using interpolation
    circular_img = interpolate_to_circular(img)

    # Save circular image
    circular_output_path = OUTPUT_DIR / f"{input_path.stem}_circular{input_path.suffix}"
    cv2.imwrite(str(circular_output_path), circular_img)
    log.info(f"Saved circular image: {circular_output_path}")

    # Apply fisheye distortion
    distorted_img = distort_image(circular_img)

    # Save fisheye image
    fisheye_output_path = OUTPUT_DIR / f"{input_path.stem}_fisheye{input_path.suffix}"
    cv2.imwrite(str(fisheye_output_path), distorted_img)
    log.info(f"Saved fisheye image: {fisheye_output_path}")


def main():
    """Main function to process all images in the input directory."""
    input_dir = Path("./")  # Set to the current directory
    input_images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")) + list(input_dir.glob("*.webp"))

    if not input_images:
        log.error("No images found in the directory!")
        return

    log.info(f"Found {len(input_images)} images to process.")

    # Use multiprocessing to process images faster
    with concurrent.futures.ProcessPoolExecutor() as executor:
        with tqdm(total=len(input_images)) as pbar:
            for _ in executor.map(process_image, input_images):
                pbar.update()


if __name__ == "__main__":
    main()
