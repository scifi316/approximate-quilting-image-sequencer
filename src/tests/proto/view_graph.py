import numpy as np
import faiss
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

# Load the descriptors from a file
descriptors = np.load('descriptors.npy').astype('float32')

# Perform PCA using Faiss to reduce to 3D
d_in = descriptors.shape[1]  # Original dimensionality (e.g., 128 for SIFT)
d_out = 3  # Target dimensionality for 3D visualization

# Create a PCA object in Faiss
pca_matrix = faiss.PCAMatrix(d_in, d_out)

# Train PCA on the descriptors
pca_matrix.train(descriptors)

# Apply the PCA transform to get 3D vectors
reduced_vectors = pca_matrix.apply_py(descriptors)

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the reduced vectors
ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], reduced_vectors[:, 2], s=5, alpha=0.7)
ax.set_title('3D Visualization of Faiss Index Vectors (PCA)')
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')

plt.show()