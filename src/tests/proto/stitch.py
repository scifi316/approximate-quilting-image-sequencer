import cv2
import numpy as np
import faiss
from pathlib import Path
import os
from collections import Counter

root_dir = Path(__file__).resolve().parents[3]

def splitImage(image, chunk_width, chunk_height):
    """Split an image into smaller chunks of the given size."""
    chunks = []
    h, w = image.shape[:2]

    for y in range(0, h, chunk_height):
        for x in range(0, w, chunk_width):
            chunk = image[y:y + chunk_height, x:x + chunk_width]
            chunks.append((x, y, chunk))
    
    return chunks

def detectFeatures(image_chunk):
    orb = cv2.ORB_create()
    gray_chunk = cv2.cvtColor(image_chunk, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray_chunk, None)
    
    return des

def matchFeatures(descriptors, faiss_index, frame_ids, expected_dimension=128):
    """Match the descriptors of a chunk to the aggregated index of MV frames."""
    if descriptors is None or len(descriptors) == 0:
        return None, None  # No features detected

    # Aggregate descriptors (e.g., using mean pooling)
    aggregated_vector = np.mean(descriptors, axis=0).astype('float32')

    # Zero-pad the vector if its dimension is smaller than the expected dimension
    if aggregated_vector.shape[0] < expected_dimension:
        padded_vector = np.zeros(expected_dimension, dtype='float32')
        padded_vector[:aggregated_vector.shape[0]] = aggregated_vector
        aggregated_vector = padded_vector

    # Ensure the vector has the correct shape for the Faiss index
    aggregated_vector = aggregated_vector.reshape(1, -1)

    # Check dimension consistency
    if aggregated_vector.shape[1] != faiss_index.d:
        print(f"Error: Aggregated vector dimension {aggregated_vector.shape[1]} does not match Faiss index dimension {faiss_index.d}.")
        return None, None

    # Perform the search in Faiss (find the 1 nearest neighbor)
    distances, indices = faiss_index.search(aggregated_vector, 1)

    # Get the frame ID of the best match
    best_match_index = indices[0][0]
    best_match_frame_id = frame_ids[best_match_index]

    return best_match_frame_id, distances[0][0]

from concurrent.futures import ThreadPoolExecutor

def processChunk(chunk_info, faiss_index, frame_ids):
    x, y, chunk = chunk_info
    descriptors = detectFeatures(chunk)
    best_match_frame_id, match_score = matchFeatures(descriptors, faiss_index, frame_ids)
    
    return (x, y, best_match_frame_id, match_score)

def parallelProcessChunks(image_chunks, faiss_index, frame_ids, num_workers=8):
    """Run feature detection and matching in parallel on all image chunks."""
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(lambda chunk: processChunk(chunk, faiss_index, frame_ids), image_chunks))
        
    return results

def getResizedFrame(frame_index, mv_frames_folder):
    if frame_index is None:
        raise ValueError("Received None as frame_index, cannot retrieve frame.")
    
    frame_filename = f"output_frames{frame_index:04d}.png"
    frame_path = f"{mv_frames_folder}/{frame_filename}"
    
    mv_frame = cv2.imread(frame_path)
    
    # Check if the frame was successfully loaded
    if mv_frame is None:
        raise FileNotFoundError(f"Frame {frame_path} not found.")
    
    # Resize the MV frame to the target chunk size
    resized_frame = cv2.resize(mv_frame, (240, 180))
    
    return resized_frame

def quiltImage(chunk_results, mv_frames_folder, target_image_shape, chunk_width=240, chunk_height=180):
    """Quilt together the best-matching frames into the target image's shape."""
    h, w = target_image_shape[:2]
    quilted_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    for x, y, match_frame_id, _ in chunk_results:
        if match_frame_id is None:
            # Skip this chunk if no valid match was found
            #print(f"Skipping chunk at ({x}, {y}) due to no valid match.")
            continue

        try:
            # Construct the path to the matched frame
            frame_path = os.path.join(mv_frames_folder, match_frame_id)
            mv_frame = cv2.imread(frame_path)
            
            # Resize the MV frame to fit the chunk size (chunk_width, chunk_height)
            resized_frame = cv2.resize(mv_frame, (chunk_width, chunk_height))
            
            # Place the resized frame into the quilted image
            quilted_image[y:y + chunk_height, x:x + chunk_width] = resized_frame
            
            #print(f"Quilted chunk at ({x}, {y}) with frame {match_frame_id}")
        
        except FileNotFoundError as e:
            print(e)
            # Optionally fill in a default color for missing frames
            quilted_image[y:y + chunk_height, x:x + chunk_width] = (0, 0, 0)  # Filling with black
    
    return quilted_image

def processTargetImage(target_image_path, faiss_index_path, frame_ids_path, mv_frames_folder, chunk_width=96, chunk_height=72):
    # Load the Faiss index
    faiss_index = faiss.read_index(faiss_index_path)
    frame_ids = np.load(frame_ids_path)
    
    # Load the target image
    target_image = cv2.imread(target_image_path)

    # Split the target image into chunks
    image_chunks = splitImage(target_image, chunk_width, chunk_height)

    # Process the chunks in parallel
    chunk_results = parallelProcessChunks(image_chunks, faiss_index, frame_ids, num_workers=8)  # Adjust num_workers as needed

    # Quilt the final image using the matched frames
    quilted_image = quiltImage(chunk_results, mv_frames_folder, target_image.shape, chunk_width, chunk_height)

    # Save and display the quilted image
    output_image_path = root_dir/f"data/images/quilted_output/quilted_frame{i:04d}.png"
    cv2.imwrite(output_image_path, quilted_image)
    #cv2.imshow('Quilted Image', quilted_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    #print(f"Quilted image saved to {output_image_path}")
    
target_image_path = root_dir/'data/images/source'
faiss_index_path = 'aggregated_faiss_index.bin'
frame_ids_path = 'frame_ids.npy'
mv_frames_folder = root_dir/'data/images/input'

# Run the main function
#processTargetImage(target_image_path, faiss_index_path, frame_ids_path, mv_frames_folder)
i = 1
for filename in sorted(os.listdir(target_image_path)):
    if filename.lower().endswith('.png'):
        target_image = os.path.join(target_image_path, filename)
        processTargetImage(target_image, faiss_index_path, frame_ids_path, mv_frames_folder)
        i += 1
        
print("Quilted all images")