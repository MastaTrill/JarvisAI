

import os
import sys
import logging
from typing import List
from PIL import Image
import numpy as np

try:
    import pandas as pd
except ImportError:
    print("pandas is required to run this script.")
    sys.exit(1)

IMAGE_DIR = "data/images"
os.makedirs(IMAGE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def generate_images(image_ids: List[str], varied: bool = True, output_dir: str = IMAGE_DIR) -> None:
    """
    Generate and save images for the given image IDs.
    Args:
        image_ids: List of image IDs to generate images for.
        varied: Whether to use varied image generation.
        output_dir: Directory to save images.
    """
    for idx, img_id in enumerate(image_ids):
        arr = generate_varied_image(idx) if varied else np.full((32, 32), 60 + idx * 50, dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        img_path = os.path.join(output_dir, f"{img_id}.png")
        try:
            img.save(img_path)
            logger.info(f"Saved image: {img_path}")
        except Exception as e:
            logger.error(f"Failed to save image {img_id}: {e}")

def generate_varied_image(idx: int) -> np.ndarray:
    """
    Generate a 32x32 grayscale image array with variation.
    Args:
        idx: Index for variation.
    Returns:
        np.ndarray: Image array.
    """
    base = 60 + (idx % 6) * 30
    arr = np.full((32, 32), base, dtype=np.uint8)
    if idx % 3 == 0:
        arr = arr + np.random.randint(0, 40, (32, 32), dtype=np.uint8)  # Add noise
    elif idx % 3 == 1:
        arr[:, ::2] = np.clip(arr[:, ::2] + 40, 0, 255)  # Add stripes
    else:
        arr = arr + np.linspace(0, 60, 32).astype(np.uint8).reshape(1, -1)  # Add gradient
    arr = np.clip(arr, 0, 255)
    return arr

def read_image_ids(csv_path: str) -> List[str]:
    """
    Read image IDs from a CSV file.
    Args:
        csv_path: Path to the CSV file.
    Returns:
        List of image IDs.
    """
    try:
        df = pd.read_csv(csv_path)
        image_ids = df["image_id"].astype(str).tolist()
        logger.info(f"Loaded {len(image_ids)} image IDs from {csv_path}")
        # Ensure all items are strings
        image_ids = [str(i) for i in image_ids if isinstance(i, str) or isinstance(i, int) or isinstance(i, float)]
        return image_ids
    except Exception as e:
        logger.error(f"Failed to read CSV {csv_path}: {e}")
        return []

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate sample images for given image IDs.")
    parser.add_argument("--csv", type=str, default="data/new_data.csv", help="Path to CSV file with image_id column.")
    parser.add_argument("--output", type=str, default=IMAGE_DIR, help="Directory to save images.")
    parser.add_argument("--simple", action="store_true", help="Use simple image generation (no variation).")
    args = parser.parse_args()

    image_ids = read_image_ids(args.csv)
    if not image_ids:
        logger.error("No image IDs found. Exiting.")
        sys.exit(1)

    logger.info("Generating sample images...")
    generate_images(image_ids, varied=not args.simple, output_dir=args.output)
    logger.info("Done.")

if __name__ == "__main__":
    main()
