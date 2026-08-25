import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import faiss

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.common import list_image_files  # noqa: E402

DESCRIPTOR_DIM = 128  # SIFT descriptor dimensionality

# See benchmarks/faiss_index_benchmark.py and benchmarks/RESULTS.md for how
# these were chosen. IVFFlat and IVFPQ need a representative sample of
# vectors before any can be added (Faiss trains cluster centroids up front),
# so they can't be filled incrementally like flat/hnsw can.
INDEX_TYPES = ("flat", "hnsw", "ivfflat", "ivfpq")
TRAINED_INDEX_TYPES = ("ivfflat", "ivfpq")
DEFAULT_INDEX_TYPE = "hnsw"

HNSW_M = 32  # neighbors per node; Faiss's own default
IVF_PQ_SUBQUANTIZERS = 16  # DESCRIPTOR_DIM (128) must be divisible by this
IVF_PQ_BITS = 8
IVF_NPROBE = 16  # inverted-list cells scanned per query; see benchmarks/RESULTS.md


def _nlistFor(num_descriptors):
    """Rule of thumb: roughly 4*sqrt(n) inverted-list cells, clamped to a
    sane range so tiny datasets (tests, small runs) don't get an
    over-partitioned index. Faiss's k-means training requires at least as
    many points as clusters, so also never exceed num_descriptors itself."""
    upper_bound = max(1, min(4096, num_descriptors))
    return max(1, min(upper_bound, int(4 * (num_descriptors ** 0.5))))


def _pqBitsFor(num_descriptors):
    """Product quantization trains 2**bits centroids per sub-quantizer, which
    requires at least that many training points. Scale bits down for small
    datasets (tests, tiny runs) instead of hard-failing; real-scale datasets
    comfortably support the full IVF_PQ_BITS."""
    if num_descriptors < 2:
        return 1
    return max(1, min(IVF_PQ_BITS, int(math.log2(num_descriptors))))


def _createIndex(index_type, dim, num_descriptors=0):
    """Factory for an (untrained, for the types that need it) Faiss index of
    the given type. num_descriptors is only used to size IVF's inverted-list
    count and is ignored by flat/hnsw."""
    if index_type == "flat":
        return faiss.IndexFlatL2(dim)
    if index_type == "hnsw":
        return faiss.IndexHNSWFlat(dim, HNSW_M)
    if index_type == "ivfflat":
        quantizer = faiss.IndexFlatL2(dim)
        return faiss.IndexIVFFlat(quantizer, dim, _nlistFor(num_descriptors))
    if index_type == "ivfpq":
        quantizer = faiss.IndexFlatL2(dim)
        return faiss.IndexIVFPQ(quantizer, dim, _nlistFor(num_descriptors),
                                 IVF_PQ_SUBQUANTIZERS, _pqBitsFor(num_descriptors))
    raise ValueError(f"Unknown Faiss index type {index_type!r}; choose from {INDEX_TYPES}.")


def buildDatabase(mv_frames_folder, output_dir=".", index_type=DEFAULT_INDEX_TYPE):
    """Build a Faiss index of individual SIFT descriptors from the MV frames,
    along with a mapping from each descriptor back to the frame it came from.

    For flat/hnsw (no training step), descriptors are added to the Faiss
    index incrementally, one frame at a time, instead of being accumulated
    in memory and concatenated at the end -- this keeps peak memory roughly
    proportional to one frame instead of the whole dataset. IVF-based types
    need a representative sample of vectors before training their cluster
    centroids, so for those, descriptors are collected across the pass and
    the index is trained and filled once at the end.
    """
    if index_type not in INDEX_TYPES:
        raise ValueError(f"Unknown Faiss index type {index_type!r}; choose from {INDEX_TYPES}.")

    needs_training = index_type in TRAINED_INDEX_TYPES

    sift = cv2.SIFT_create()
    faiss_index = None if needs_training else _createIndex(index_type, DESCRIPTOR_DIM)
    pending_descriptors = [] if needs_training else None

    frame_ids = []
    frame_to_descriptor_indices = []

    for filename in list_image_files(mv_frames_folder):
        frame_path = os.path.join(mv_frames_folder, filename)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Skipping {filename}: could not read image.")
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, descriptors = sift.detectAndCompute(gray_frame, None)

        if descriptors is None or len(descriptors) == 0:
            print(f"Skipping {filename}: no features detected.")
            continue

        if descriptors.shape[1] != DESCRIPTOR_DIM:
            print(f"Skipping {filename}: descriptor dimension {descriptors.shape[1]} "
                  f"does not match expected dimension {DESCRIPTOR_DIM}.")
            continue

        # Use the frame's about-to-be-assigned position in frame_ids as its
        # frame index, not its position in the source directory listing --
        # otherwise the mapping desyncs from frame_ids as soon as any frame
        # is skipped above.
        frame_index = len(frame_ids)
        descriptors = descriptors.astype('float32')
        if needs_training:
            pending_descriptors.append(descriptors)
        else:
            faiss_index.add(descriptors)
        frame_to_descriptor_indices.extend([frame_index] * len(descriptors))
        frame_ids.append(filename)

    if needs_training:
        all_descriptors = (
            np.vstack(pending_descriptors) if pending_descriptors
            else np.empty((0, DESCRIPTOR_DIM), dtype='float32')
        )
        faiss_index = _createIndex(index_type, DESCRIPTOR_DIM, num_descriptors=len(all_descriptors))
        if len(all_descriptors) > 0:
            faiss_index.train(all_descriptors)
            faiss_index.add(all_descriptors)
            # IVF indexes default to nprobe=1 (scanning a single inverted-list
            # cell per query), which tanks recall. nprobe is serialized with
            # the index, so setting it once here is enough for stitch.py's
            # faiss.read_index() to pick it up automatically.
            if hasattr(faiss_index, "nprobe"):
                faiss_index.nprobe = IVF_NPROBE

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(faiss_index, str(output_dir / 'individual_descriptors_faiss_index.bin'))
    np.save(output_dir / 'frame_ids.npy', np.array(frame_ids))
    np.save(output_dir / 'frame_to_descriptor_indices.npy', np.array(frame_to_descriptor_indices))

    return faiss_index, frame_ids, frame_to_descriptor_indices


if __name__ == "__main__":
    mv_frames_folder = ROOT_DIR / 'data/images/input'
    index_type = os.environ.get('FAISS_INDEX_TYPE', DEFAULT_INDEX_TYPE)
    buildDatabase(mv_frames_folder, output_dir=ROOT_DIR, index_type=index_type)
    print(f"Faiss index ({index_type}) created and saved successfully.")
