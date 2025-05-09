import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology, segmentation
from sklearn.cluster import KMeans
from tqdm import tqdm


def isolate_person(depth):
    depth_blur = cv2.medianBlur(depth, 3) # Just in case you need to blur later

    h, w = depth.shape
    depth_no_floor = depth_blur

    # Step 2: Threshold by depth percentile (ignoring floor)
    valid_depths = depth_no_floor[depth_no_floor > 0]
    if len(valid_depths) < 50:
        print("⚠️ Not enough valid depths for clustering.")
        return depth, np.zeros_like(depth, dtype=np.uint8), np.zeros_like(depth)
    
    depth_thresh = np.percentile(valid_depths, 20)
    mask = (depth_no_floor > 0) & (depth_no_floor < depth_thresh)

    # Step 3: Connected components
    mask_uint8 = (mask * 255).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8)

    # Step 4: Filter blobs
    person_mask = np.zeros_like(mask_uint8)
    for i in range(1, n_labels):
        x, y, w, h, area = stats[i]
        aspect_ratio = h / (w + 1e-5)
        if area > 15:
            #print("✅ Found blob", i, "with area", area, "and aspect ratio", aspect_ratio) 
            person_mask[labels == i] = 255

    # Step 5: Masked result
    h, w = person_mask.shape
    floor_cutoff = int(h * 0.79)
    side_cutoff = int(w * 0.02)
    person_mask[:, :side_cutoff] = 0  # Set left region to 0 to ignore it
    person_mask[floor_cutoff:, :] = 0  # Set floor region to 0 to ignore it
    isolated = np.where(person_mask == 255, depth, 0)
    return depth, person_mask, isolated

def isolate_train_data():
    root_dir = "pose-dataset/train"  # Replace with your root directory
    out_dir = "pose-dataset/pose-depth-masks"  # Replace with your output directory
    for subdir, _, files in os.walk(root_dir):
        mask_list = []

        # Filter files (e.g. include only isolation masks by name or extension)
        for file in tqdm(sorted(files), desc="files"):  # sort to keep consistent order
            if file.endswith(('npy')) and '_2025' in file:  # Adjust this condition as needed
                file_path = os.path.join(subdir, file)
                depth_image = np.load(file_path)
                depth_image = depth_image[:384, :]  # Only the top 384 rows
                depth_image, person_mask, isolated = isolate_person(depth_image)
                mask_list.append(isolated)

        if mask_list:
            stacked_masks = np.stack(mask_list)  # Shape: (N, H, W)
            # Output filename based on folder name
            subdir_name = os.path.basename(subdir.rstrip("/"))
            output_path = os.path.join(out_dir, f"{subdir_name}_isolation_masks.npy")
            np.save(output_path, stacked_masks)
            print(f"Saved {len(mask_list)} masks to: {output_path}")
def display_all(): 
    # ---------- CONFIGURE THIS ----------
    folder_path = "pose-dataset/train/hip-forward/"
    # ------------------------------------

    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])
    i = 0
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        depth = np.load(file_path).astype(np.float32)
        depth = depth[:384, :]  # Only the top 384 rows
        #depth = cv2.resize(depth, (320, 240), interpolation=cv2.INTER_LINEAR)
        original, mask, isolated = isolate_person(depth)
        # Plot
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].imshow(original, cmap='viridis')
        axs[0].set_title(f"Original: {file_name}")
        axs[1].imshow(isolated, cmap='gray')
        axs[1].set_title("Mask")
        plt.show()
        if file_name == "side_2025-04-29_14-02-48.npy":
            print("saving best")
            np.save("example_masked_depth.npy", isolated)
        input("Press Enter to continue to the next image...")

def isolate_live_feed(folder_path): 
    print("Isolating live feed")
    masks = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if file_path.endswith('.npy'): 
            depth_image = np.load(file_path)
            depth_image = depth_image[:384, :]  # Only the top 384 rows
            depth_image, person_mask, isolated = isolate_person(depth_image)
            masks.append(isolated)
    np.stack(masks)
    return masks 
            

            


