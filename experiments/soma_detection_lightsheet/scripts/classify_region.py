import os
from pathlib import Path
from cloudvolume.lib import Bbox
import numpy as np
import math
import itertools
import time
import matplotlib.pyplot as plt
from cloudvolume import CloudVolume
from brainlit.algorithms.detect_somas import find_somas

viz = False
mip = 0

cwd = Path(os.path.abspath(__file__))
exp_dir = cwd.parents[1]
data_dir = os.path.join(exp_dir, "data")

# brains = [1]

# s3 = boto3.resource("s3")
# bucket = s3.Bucket("open-neurodata")

# brain = 1
# brain_name = f"brain{brain}"

# brain_dir = os.path.join(data_dir, brain_name)
results_dir = os.path.join(data_dir, "results")
# tmp_coords_path = os.path.join(results_dir, "tmp_coords.npy")

# brain_prefix = f"brainlit/{brain_name}"
# segments_prefix = f"brainlit/{brain_name}_segments"
# somas_prefix = f"brainlit/{brain_name}_octant"

brain_url = "precomputed://https://dlab-colm.neurodata.io/2021_03_10/Mouse10-Cohort1/ch561-corrected"
# segments_url = f"s3://open-neurodata/{segments_prefix}"

# ngl_sess = NeuroglancerSession(mip=1, url=brain_url, url_segments=segments_url)
cv = CloudVolume(brain_url, mip=mip, fill_missing=True)
res = cv.scales[mip]["resolution"]
# res = ngl_sess.cv_segments.scales[ngl_sess.mip]["resolution"]

min_region_lims = [5531640, 1357800, 2000000]
max_region_lims = [9073080, 5315440, 2800000]
step = [300000, 300000, 300000]
# brain_lims = [6, 8, 10]
# step = [2, 1, 3]

region = [min_region_lims, max_region_lims]

N = [range(region[0][i] // step[i], region[1][i] // step[i] + (1 if np.abs(region[0][i] - region[1][i]) % d != 0 else 0)) for i, d in enumerate(step)]
print(N)

_iter_discrete_coords = itertools.product(N[0], N[1], N[2])

def discrete_to_spatial(discrete_x, discrete_y, discrete_z):
    discrete_coords = [discrete_x, discrete_y, discrete_z]
    print(discrete_coords)
    return [[k * step[i] for i, k in enumerate(discrete_coords)], [(k + 1) * step[i] for i, k in enumerate(discrete_coords)]]

for i, volume_coords in enumerate(itertools.starmap(discrete_to_spatial, _iter_discrete_coords)):
    print(volume_coords)
    volume_id = f"{volume_coords[0][0]}_{volume_coords[0][1]}_{volume_coords[0][2]}_{volume_coords[1][0]}_{volume_coords[1][1]}_{volume_coords[1][2]}"
    
    volume_min_vox = np.round(np.divide(np.array(volume_coords[0]), res)).astype(int)
    volume_max_vox = np.round(np.divide(np.array(volume_coords[1]), res)).astype(int)
    
    bbox = Bbox(volume_min_vox, volume_max_vox)
    print(f"============\nPulling {i}-th volume, bbox={bbox}...", end="", flush=True)
    t0 = time.time()
    volume = cv.download(bbox, mip=mip).squeeze()
    t = time.time()
    dt = np.around(t - t0, decimals=3)
    print(f"done in {dt}s")
    
    try:
        print("Looking for somas...", end="", flush=True)
        t0 = time.time()
        label, rel_pred_centroids, mask = find_somas(volume, res)
        t = time.time()
        dt = np.around(t - t0, decimals=3)
        print(f"done in {dt}s")
    except ValueError:
        print(f"failed")
    else:
        print(f"Found {len(rel_pred_centroids)} somas")
        pred_centroids = np.array([np.multiply(volume_min_vox + c, res) for c in rel_pred_centroids])
        
        _, axes = plt.subplots(1, 2)
        
        ax = axes[0]
        vol_proj = np.amax(volume, axis=2)
        ax.imshow(vol_proj, cmap="gray", origin="lower")
        if len(rel_pred_centroids) > 0:
            ax.scatter(rel_pred_centroids[:, 1], rel_pred_centroids[:, 0], c="b", alpha=.5)
        
        ax = axes[1]
        mask_proj = np.amax(mask, axis=2)
        ax.imshow(mask_proj, cmap="jet", vmin=0, origin="lower")
        
        plt.savefig(os.path.join(results_dir, f"{volume_id}.jpg"))
        
        # volume_key = f"{somas_prefix}/{volume_id}"
        
        # np.save(tmp_coords_path, pred_centroids, allow_pickle=True)
        
        # print(f"Uploading coordinates to S3...", end="", flush=True)
        # t0 = time.time()
        # bucket.upload_file(tmp_coords_path, volume_key)
        # t = time.time()
        # dt = np.around(t - t0, decimals=3)
        # print(f"done in {dt}s")