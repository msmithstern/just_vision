import numpy as np
import matplotlib.pyplot as plt

def load_data():
    """
    Load from depths.npy, segms.npy, joints2d.npy
    """
    depth = np.load("depth.npy")
    segm = np.load("segm.npy")
    joints2d = np.load("joints2d.npy")
    return depth, segm, joints2d

depth, segm, joints2d = load_data()

# Plot each depth, segmentation map, and joint locations
for i in range(depth.shape[0]):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot depth
    axes[0].imshow(depth[i], cmap='gray')
    axes[0].set_title("Depth")
    axes[0].axis('off')
    
    # Plot segmentation map
    axes[1].imshow(segm[i], cmap='viridis')
    axes[1].set_title("Segmentation Map")
    axes[1].axis('off')
    
    # Overlay joint locations on depth
    plt.figure(figsize=(5, 5))
    plt.imshow(depth[i], cmap='gray')
    plt.scatter(joints2d[i, :, 0], joints2d[i, :, 1], c='red', s=10, label='Joints')
    plt.title("Depth with Joint Locations")
    plt.axis('off')
    plt.legend()
    
    plt.show()
