import os

import cv2
import numpy as np

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')


def list_image_files(folder):
    """Return sorted image filenames (png/jpg/jpeg) in the given folder."""
    return sorted(
        filename for filename in os.listdir(folder)
        if filename.lower().endswith(IMAGE_EXTENSIONS)
    )


def tileDescriptorDim(thumb_size, channels=3):
    """Dimensionality of the descriptor computeTileDescriptors produces."""
    return thumb_size * thumb_size * channels


def computeTileDescriptors(image, chunk_width, chunk_height, thumb_size=4):
    """Compute one dense descriptor per grid tile of the image, by area-
    averaging each tile down to a thumb_size x thumb_size thumbnail.

    This exists as an alternative to sparse keypoint detectors (SIFT) for
    fine-grained quilting. SIFT keypoints are sparse and clustered around
    texture/edges -- they don't scale down with tile size, so shrinking the
    tile grid mostly produces tiles with zero keypoints to match on (see
    benchmarks/RESULTS.md). A dense per-tile descriptor guarantees every
    tile gets exactly one descriptor regardless of its content, which is
    what makes finer-grained quilting viable at all.

    The whole grid is computed in one vectorized pass -- a single
    cv2.resize(..., INTER_AREA) call (which computes an area-weighted
    average per output pixel, i.e. exactly the per-tile mean color/
    luminance signature) followed by a reshape -- instead of a Python loop
    over tiles, so cost is dominated by one resize regardless of how many
    tiles that produces.

    chunk_width/chunk_height must evenly divide the image's width/height;
    this raises ValueError otherwise rather than silently cropping or
    padding, since a partial edge tile can't be pooled into a fixed-size
    thumbnail without inconsistent normalization relative to full tiles.

    Returns (descriptors, num_cols, num_rows): descriptors has shape
    (num_rows * num_cols, tileDescriptorDim(thumb_size, channels)), in the
    same row-major order (y outer, x inner) as splitImage's chunk grid.
    """
    h, w = image.shape[:2]
    if w % chunk_width != 0 or h % chunk_height != 0:
        raise ValueError(
            f"Image size {w}x{h} isn't evenly divisible by chunk size "
            f"{chunk_width}x{chunk_height}."
        )

    num_cols = w // chunk_width
    num_rows = h // chunk_height

    resized = cv2.resize(image, (num_cols * thumb_size, num_rows * thumb_size), interpolation=cv2.INTER_AREA)
    if resized.ndim == 2:
        resized = resized[..., np.newaxis]
    channels = resized.shape[2]

    tiles = resized.reshape(num_rows, thumb_size, num_cols, thumb_size, channels)
    tiles = tiles.transpose(0, 2, 1, 3, 4)  # (num_rows, num_cols, thumb_size, thumb_size, channels)
    descriptors = tiles.reshape(num_rows * num_cols, thumb_size * thumb_size * channels).astype('float32')

    return descriptors, num_cols, num_rows
