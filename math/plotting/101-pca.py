#!/usr/bin/env python3
"""
This module contains code to perform 3D visualization of the Iris Dataset
using Principal Component Analysis (PCA).
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Load the PCA dataset
lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

# --- Your code here ---

# Initialize the figure
fig = plt.figure(figsize=(6.4, 4.8))

# Add a 3D subplot axis
ax = fig.add_subplot(111, projection='3d')

# Extract x, y, and z coordinates from the 3 dimensions of the reduced data
x = pca_data[:, 0]
y = pca_data[:, 1]
z = pca_data[:, 2]

# Plot the 3D scatter graph with plasma colormap mapped to the species labels
scatter = ax.scatter(x, y, z, c=labels, cmap='plasma')

# Add the required title and axis labels
ax.set_title('PCA of Iris Dataset')
ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')

# Display the final visualization
plt.show()
