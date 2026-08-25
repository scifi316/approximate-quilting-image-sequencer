import os
import re

import cv2
import numpy as np

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')

_DIGIT_RUN = re.compile(r'(\d+)')


def _naturalSortKey(filename):
    """Sort key that orders embedded numbers by value, not character-by-
    character -- plain lexicographic sort would order "frame10.png" before
    "frame2.png" (since '1' < '2'), which silently scrambles playback order
    for any frame sequence whose filenames aren't zero-padded to a fixed
    width. Zero-pads each digit run to a large fixed width instead of
    parsing it to int, so the key stays all-strings (safe to sort/compare
    even if folder contents mix differently-structured filenames) while
    still comparing numeric runs by value.
    """
    return [part.zfill(20) if part.isdigit() else part for part in _DIGIT_RUN.split(filename)]


def list_image_files(folder):
    """Return image filenames (png/jpg/jpeg) in the given folder, in natural
    (numeric-aware) sorted order -- see _naturalSortKey."""
    return sorted(
        (filename for filename in os.listdir(folder) if filename.lower().endswith(IMAGE_EXTENSIONS)),
        key=_naturalSortKey,
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
