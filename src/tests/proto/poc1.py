import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

root_dir = Path(__file__).resolve().parents[3]  # src/tests/proto --> $main dir

orb = cv2.ORB_create()

all_discriptors = []
keypoint_metadata = []

for filename in (os.listdir(str(root_dir/'data/images/input'))):
    if filename.lower().endswith(('png', 'jpg', 'jpeg')):
        image_path = root_dir/'data/images/input'/filename
        image = cv2.imread(str(image_path))
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        keypoints, descriptors = orb.detectAndCompute(gray_image, None)
        
        if descriptors is not None:
            all_discriptors.append(descriptors)
            keypoint_coords = np.array([kp.pt for kp in keypoints], dtype=np.float32)
            keypoint_metadata.append((filename, keypoint_coords))
            # print(f"Processed image: {filename}: {len(keypoints)} keypoints")
        else:
            print(f"Processed image: {filename}: No keypoints found")

if all_discriptors:
    descriptors_array = np.vstack(all_discriptors).astype(np.float32)
    print(f"Descriptors array shape: {descriptors_array.shape}")
    
    np.save('descriptors.npy', descriptors_array)
    # np.save('keypoint_metadata.npy', keypoint_metadata)
else:
    print("No descriptors found\n")
    
# image_path = root_dir/'data/images/input/test-img1.png' 
# image = cv2.imread(str(image_path))

# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# keypoints, descriptors = orb.detectAndCompute(gray_image, None)

# keypoint_coords = np.array([kp.pt for kp in keypoints], dtype=np.float32)

# image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# plt.figure(figsize=(12, 6))
# plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
# plt.title('Image with SIFT Keypoints')
# plt.axis('off')
# plt.show()

# if descriptors is not None:
#     np.save('discriptors.npy', descriptors)
#     np.save('keypoint_coords.npy', keypoint_coords)
    
#     print(f"Number of keypoints detected: {len(keypoints)}")
#     print(f"Descriptor of array shape: {descriptors.shape}")
#     print(f"Keypoints coordinates array shape: {keypoint_coords.shape}")
# else:
#     print("No descriptors found\n")

import faiss

index = faiss.IndexFlatL2(descriptors_array.shape[1])

index.add(descriptors_array)

faiss.write_index(index, 'index_faiss.bin')
print(f"Faiss index created with {index.ntotal} vectors.")