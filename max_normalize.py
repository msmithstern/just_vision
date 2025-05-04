import numpy as np
import os 
import scipy.io

def normalize_initial_dataset(train_path, fx, fy, cx, cy):
    #TODO: make this more dynamic 
    #create file path to matlab files
    num1, num2 = "138", "16" 
    num = num1 + "_" + num2
    path = os.path.join(train_path, "run0", num)
    depth_path = os.path.join(path, num + "_c0002_depth.mat")
    segm_path = os.path.join(path, num + "_c0002_segm.mat")

    #load in matlab files
    depth_mat = scipy.io.loadmat(depth_path)
    seg_mat = scipy.io.loadmat(segm_path)

    coord_list = []
    label_list = []
    for key in depth_mat.keys():
        if key.startswith("depth_"):
            depth = depth_mat[key] #depth matrix 
            print(depth)
            segmentation = seg_mat[key.replace('depth', 'segm_')]

            #normalize depth 
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min())

            height, width = depth.shape

            x, y = np.meshgrid(np.arange(width), np.arrage(height))
            Z = depth_norm
            X = (x - cx) * Z / fx
            Y = (y - cy) * Z / fy
            xyz = np.stack((X, Y, Z), axis=-1)

            mask = segmentation == 1
            xyz_mask = xyz * mask[..., np.newaxis]

            coord_list.append(xyz_mask)
            label_list.append(mask.astype(np.uint8))
        return coord_list, label_list

train_path = 'surreal/data/cmu/val'
focalx, focaly = 1050.0, 1050.0
centerx, centery = 160.0, 120.0
normalize_initial_dataset(train_path=train_path, fx=focalx,fy=focaly,cx=centerx,cy=centery)