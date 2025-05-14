import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology, segmentation 
from sklearn.cluster import KMeans
from tqdm import tqdm

def isolate_person(depth):
    """
    This is a function that isolates a person with a threshold.

    Parameters: depth - a depth numpy object
    Returns: tuple of depth numpy object, mask with person and isolated person
    """
    #blur image
    depth_blur = cv2.medianBlur(depth, 3)

    h, w = depth.shape
    # threshold for valid depths 
    valid_depths = depth_blur[depth_blur > 0]
    if len(valid_depths) < 50:
        print("Not enough valid depths for clustering")
        return depth, np.zeros_like(depth, dtype=np.uint8), np.zeros_like(depth)
    depth_thresh = np.percentile(valid_depths,20)
    #mask only close by depths
    mask = (depth_blur > 0) & (depth_blur < depth_thresh)

    #use connected components to get depth blobs
    mask_unit8 = (mask * 255).astype(np.unit8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_unit8)

    #filter blobs
    person_mask = np.zeros_like(mask_unit8)
    for i in range(1, n_labels):
        #use area thresholding
        x, y, w, h, area = stats[i]
        if area > 15:
            person_mask[labels == i] = 255
    
    #adjust final masked result
    h, w = person_mask.shape
    floor_cutoff = int(h * 0.79)
    side_cutoff = int(w * 0.02)
    #manually cut off the floor and the sides to reduce noise
    person_mask[:, :side_cutoff] = 0
    person_mask[floor_cutoff:, :] = 0
    isolated = np.where(person_mask == 255, depth, 0)
    return depth, person_mask, isolated

def isolate_train_data():
    """
    This function isolates a person in training data. 

    Parameters: None
    Returns: None
    """
    root_dir = "pose-dataset/train"
    out_dir = "pose-dataset/pose-depth-masks"
    for subdir, _, files in os.walk(root_dir):
        mask_list = []
        #iterate through training data
        for file in tqdm(sorted(files), desc="files"):
            if file.endswith(('npy')) and '_2025' in file:
                file_path = os.path.join(subdir, file)
                depth_image = np.load(file_path)
                depth_image = depth_image[: 384, :] #only top rows
                depth_image, _, isolated = isolate_person(depth_image)
                mask_list.append(isolated)

        if mask_list:
            stacked_masks = np.stack(mask_list)
            #output file name based on folder name
            subdir_name = os.path.basename(subdir.rstrip("/"))
            output_path = os.path.join(out_dir, f"{subdir_name}_isolation_masks.npy")
            np.save(output_path, stacked_masks)
            print(f"Saved {len(mask_list)} masks to: {output_path}")

def display_all():
    """
    This function displays all the masks as a plot
    
    Parameters: None
    Returns: None
    """
    folder_path = "pose-dataset/train/hip-forward/"

    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])
    i = 0
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        depth = np.load(file_path).astype(np.float32)
        #crop image to match SURREAL ratio
        depth = depth [:384, :]
        original, mask, isolated = isolate_person(depth)
        # plot
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].imshow(original, cmap='viridis')
        axs[0].set_title(f"Original: {file_name}")
        axs[1].imshow(isolated, cmap='gray')
        axs[1].set_title("Mask")
        plt.show()
        if file_name == "side_2025-04-29_14-02-48.npy":
            print("saving best")
            np.save("example_masked_depth.npy", isolated)
        input("press enter")

def isolate_live_feed(folder_path):
    """
    This function isolates the live feed for our just_dance.py file.
    
    Parameters: None
    Returns: None
    """
    print("isolating live feed")
    masks = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if file_path.endswith('.npy'):
            depth_image = np.load(file_path)
            # croop to match SURREAL ratio 
            depth_image = depth_image[:384, :]
            depth_image, person_mask, isolated = isolate_person(depth_image)
            masks.append(isolated)
    np.stack(masks)
    return masks 