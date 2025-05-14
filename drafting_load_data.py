import os
import numpy as np
import scipy.io

def load_triplet(depth_path, segm_path, info_path):
    """
    This function downloads the SURREAL dataset into separate .npy files 

    Parameters: 
        depth_path - path to depth file
        segm_path - path to segmentation file
        info_path - path to info file

    Returns:
        tuple of depth array, segmentation array, and joints array
    """
    try: 
        #load mat files 
        depth_mat = scipy.io.loadmat(depth_path) #returns dict
        segm_mat = scipy.io.loadmat(segm_path)
        info_mat = scipy.io.loadmat(info_path)

        #put them into arrays since loadmat returns dict
        depth_arr = [v for k, v in depth_mat.items() if "depth" in k.lower()][0]
        segm_arr = [v for k, v in segm_mat.items() if "segm" in k.lower()][0]
        joints2d = info_mat.get('joints2D', None)

        #check that joints2d is in info file
        if joints2d is None or joints2d.shape[0] != 2:
            raise ValueError("Invalid or missing joints2d", joints2d.shape[0])
        if depth_arr.ndim == 2:
            depth_arr = depth_arr[np.newaxis, ...]
        if segm_arr.ndim == 2:
            segm_arr = segm_arr[np.newaxis, ...]
        
        #made sure shape is (n, 24, 2)
        joints2d = np.transpose(joints2d, (2,1,0))

        #match shortest time
        T = depth_arr.shape[0]
        if joints2d.shape[0] < T:
            raise ValueError("frames do not align")
        return depth_arr, segm_arr, joints2d[:T]
    except Exception as e:
        print(f"skipping due to error {e}")
        return None, None, None


def find_valid_triplets(root_dir):
    """
    This function returns a list of valid triplets of depth/segm/info matlab files. 

    Parameters:
        root_dir - root directory 

    Returns: 
        valid triplets in a python list
    """
    files = os.listdir(root_dir)
    bases = set()

    #sort triplet by bases
    for f in files:
        if f.endswith("_depth.mat"):
            bases.add(f.replace("_depth.mat", ""))
        if f.endswith("_segm.mat"):
            bases.add(f.replace("_segm.mat"), "")
        if f.endswith("_info.mat"):
            bases.add(f.replace("_info.mat"), "")
    
    valid_triplets = []
    for base in sorted(bases):
        dpath = os.path.join(root_dir, base + "_depth.mat")
        spath = os.path.join(root_dir, base + "_segm.mat")
        ipath = os.path.join(root_dir, base + "_info.mat")
        if all(os.path.exists(p) for p in [dpath, spath, ipath]):
            valid_triplets.append((dpath, spath, ipath))

    print(f"Found {len(valid_triplets)} valid triplets")
    return valid_triplets

def load_all_triplets(root_dir):
    """
    This function loads all the triplets using find_valid_triplets. 

    Parameters:
        root_dir - root directory 

    Returns: 
        triplets as numpy files 
    """
    triplets = find_valid_triplets(root_dir)
    if not triplets:
        raise RuntimeError("uh oh")
    depth_all, segm_all, joints3d_all = [], [], [] 

    for dpath, spath, ipath in triplets:
        depth, segm, joints3d = load_triplet(dpath, spath, ipath)
        if depth is None: #skip bad samples
            continue 
        depth_all.append(depth)
        segm_all.append(segm)
        joints3d_all.append(joints3d)
    
    if not depth_all:
        raise RuntimeError("uh oh no data tuples found")
    
    depth_all = np.concatenate(depth_all, axis=0)
    segm_all = np.concatenate(segm_all, axis=0)
    joints3d_all = np.concatenate(joints3d_all, axis=0)
    return depth_all, segm_all, joints3d_all

if __name__ == "__main__":
    root_dir = "SURREAL"
    depth_all, segm_all, joints_3d_all = load_all_triplets(root_dir)

    #save to avoid rerunning 
    np.save("depth.npy", depth_all)
    np.save("segm.npy", segm_all)
    np.save("joints2d.npy", joints_3d_all)
    print("Saved depth.npy, segm.npy,joints2d.npy")