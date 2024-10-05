import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import faiss

root_dir = Path(__file__).resolve().parents[3]  # src/tests/proto --> $main dir

def buildDatabase(mv_frames_folder):
    """Builds the master database of individual descriptors from MV frames without aggregation."""
    descriptors_list = []
    frame_to_descriptor_indices = []  # To keep track of which descriptors belong to which frames
    frame_ids = []

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    for frame_idx, filename in enumerate(sorted(os.listdir(mv_frames_folder))):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            frame_path = os.path.join(mv_frames_folder, filename)
            frame = cv2.imread(frame_path)

            # Convert to grayscale and detect features
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = sift.detectAndCompute(gray_frame, None)

            if descriptors is not None and len(descriptors) > 0:
                # Ensure all descriptors have the correct dimensionality
                descriptor_dim = descriptors.shape[1]
                if descriptor_dim != 128:
                    print(f"Skipping {filename}: Descriptor dimension {descriptor_dim} does not match expected dimension 128.")
                    continue

                # Append all descriptors to the list
                descriptors_list.append(descriptors)

                # Track which frame these descriptors belong to
                frame_to_descriptor_indices.extend([frame_idx] * len(descriptors))

                # Store the frame ID if not already stored (one per frame)
                if frame_idx >= len(frame_ids):
                    frame_ids.append(filename)

    # Concatenate all descriptors into a single NumPy array
    all_descriptors = np.vstack(descriptors_list).astype('float32')

    # Create the Faiss index for all individual descriptors
    faiss_index = faiss.IndexFlatL2(all_descriptors.shape[1])  # Assuming descriptors are 128-dimensional
    faiss_index.add(all_descriptors)

    # Save the Faiss index, frame IDs, and descriptor-to-frame mapping
    faiss.write_index(faiss_index, 'individual_descriptors_faiss_index.bin')
    np.save('frame_ids.npy', np.array(frame_ids))
    np.save('frame_to_descriptor_indices.npy', np.array(frame_to_descriptor_indices))

    return faiss_index, frame_ids, frame_to_descriptor_indices

mv_frames_folder = root_dir/'data/images/input'
master_database, frame_ids, frame_to_descriptor_indices = buildDatabase(mv_frames_folder)

# Create a Faiss index (using L2 distance for simplicity)
try:
    # Create the Faiss index with the correct dimensionality
    faiss_index = faiss.IndexFlatL2(master_database.shape[1])
    
    # Add the master database to the Faiss index
    faiss_index.add(master_database)
    
    # Save the Faiss index
    faiss.write_index(faiss_index, 'aggregated_faiss_index.bin')
    print("Faiss index created and saved successfully.")
except Exception as e:
    print(f"Error during index creation: {e}")