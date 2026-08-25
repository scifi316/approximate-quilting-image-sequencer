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


def buildDatabase(mv_frames_folder, output_dir="."):
    """Build a Faiss index of individual SIFT descriptors from the MV frames,
    along with a mapping from each descriptor back to the frame it came from.

    Descriptors are added to the Faiss index incrementally, one frame at a
    time, instead of being accumulated in memory and concatenated at the end
    -- this keeps peak memory roughly proportional to one frame instead of
    the whole dataset.
    """
    sift = cv2.SIFT_create()
    faiss_index = faiss.IndexFlatL2(DESCRIPTOR_DIM)

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
        faiss_index.add(descriptors.astype('float32'))
        frame_to_descriptor_indices.extend([frame_index] * len(descriptors))
        frame_ids.append(filename)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(faiss_index, str(output_dir / 'individual_descriptors_faiss_index.bin'))
    np.save(output_dir / 'frame_ids.npy', np.array(frame_ids))
    np.save(output_dir / 'frame_to_descriptor_indices.npy', np.array(frame_to_descriptor_indices))

    return faiss_index, frame_ids, frame_to_descriptor_indices


if __name__ == "__main__":
    mv_frames_folder = ROOT_DIR / 'data/images/input'
    buildDatabase(mv_frames_folder, output_dir=ROOT_DIR)
    print("Faiss index created and saved successfully.")
