import numpy as np
import imageio

import matplotlib.pyplot as plt

# Load the depth image
depth_image_path = 'pose-dataset/train/y/y_2025-04-29_14-13-03.npy'  # Replace with the actual path to your depth image
depth_image = np.load(depth_image_path)

# Plot the depth image using viridis colormap
plt.imshow(depth_image, cmap='gray')
plt.colorbar(label='Depth')
plt.title('Depth Image')
plt.axis('off')
plt.show()