from os.path import join
import boto3
import os
from pathlib import Path
from cloudvolume.lib import Bbox
import numpy as np
from brainlit.utils.session import NeuroglancerSession
import math
import itertools
import time

from skimage import (
    filters,
    morphology,
)

viz = False
mip = 1

cwd = Path(os.path.abspath(__file__))
exp_dir = cwd.parents[1]
data_dir = os.path.join(exp_dir, "data")

brains = [1]

s3 = boto3.resource("s3")
bucket = s3.Bucket("open-neurodata")

def contains_somas(volume):
    out = volume.copy()

    t = filters.threshold_otsu(out)
    out = out > t

    clean_selem = morphology.octahedron(2)
    cclean_selem = morphology.octahedron(1)
    out = morphology.erosion(out, clean_selem)
    out = morphology.erosion(out, cclean_selem)

    out, labels = morphology.label(out, background=0, return_num=True)
    for label in np.arange(1, labels + 1):
        A = np.sum(out.flatten() == label)
        if A < 100:
            out[out == label] = 0

    labels, m = morphology.label(out, background=0, return_num=True)
    rel_centroids = np.zeros((m, 3))
    for i, c in enumerate(range(1, m + 1)):
        ids = np.where(labels == c)
        rel_centroids[i] = np.round([np.mean(u) for u in ids]).astype(int)

    label = 0 if (m == 0 or m >= 10) else 1

    return label, rel_centroids, out

brain = 1
brain_name = f"brain{brain}"

brain_dir = os.path.join(data_dir, brain_name)
results_dir = os.path.join(brain_dir, "results")
tmp_coords_path = os.path.join(results_dir, "tmp_coords.npy")

brain_prefix = f"brainlit/{brain_name}"
segments_prefix = f"brainlit/{brain_name}_segments"
somas_prefix = f"brainlit/{brain_name}_octant"

brain_url = f"s3://open-neurodata/{brain_prefix}"
segments_url = f"s3://open-neurodata/{segments_prefix}"

ngl_sess = NeuroglancerSession(mip=1, url=brain_url, url_segments=segments_url)
res = ngl_sess.cv_segments.scales[ngl_sess.mip]["resolution"]

brain_lims = [10095672, 7793047, 13157636]
step = [100000, 100000, 100000]
# brain_lims = [6, 8, 10]
# step = [2, 1, 3]

octant = [[0, 0, 0], [math.ceil(lim / 2) for lim in brain_lims]]

N = [range(octant[0][i], octant[1][i] // step[i] + (1 if brain_lims[i] % d != 0 else 0)) for i, d in enumerate(step)]

_iter_discrete_coords = itertools.product(N[0], N[1], N[2])

def discrete_to_spatial(discrete_x, discrete_y, discrete_z):
    discrete_coords = [discrete_x, discrete_y, discrete_z]
    return [[k * step[i] for i, k in enumerate(discrete_coords)], [(k + 1) * step[i] for i, k in enumerate(discrete_coords)]]


for i, volume_coords in enumerate(itertools.starmap(discrete_to_spatial, _iter_discrete_coords)):
    print(volume_coords)
    volume_id = f"{volume_coords[0][0]}_{volume_coords[0][1]}_{volume_coords[0][2]}_{volume_coords[1][0]}_{volume_coords[1][1]}_{volume_coords[1][2]}"
    
    volume_min_vox = np.round(np.divide(np.array(volume_coords[0]), res)).astype(int)
    volume_max_vox = np.round(np.divide(np.array(volume_coords[1]), res)).astype(int)
    
    bbox = Bbox(volume_min_vox, volume_max_vox)
    print(f"============\nPulling {i}-th volume, bbox={bbox}...", end="", flush=True)
    t0 = time.time()
    volume = ngl_sess.pull_bounds_img(bbox)
    t = time.time()
    dt = np.around(t - t0, decimals=3)
    print(f"done in {dt}s")
    
    try:
        print("Looking for somas...", end="", flush=True)
        t0 = time.time()
        label, rel_pred_centroids, _ = contains_somas(volume)
        t = time.time()
        dt = np.around(t - t0, decimals=3)
        print(f"done in {dt}s")
    except ValueError:
        print(f"failed")
    else:
        print(f"Found {len(rel_pred_centroids)} somas")
        pred_centroids = np.array([np.multiply(volume_min_vox + c, res) for c in rel_pred_centroids])
        
        volume_key = f"{somas_prefix}/{volume_id}"
        
        np.save(tmp_coords_path, pred_centroids, allow_pickle=True)
        
        print(f"Uploading coordinates to S3...", end="", flush=True)
        t0 = time.time()
        bucket.upload_file(tmp_coords_path, volume_key)
        t = time.time()
        dt = np.around(t - t0, decimals=3)
        print(f"done in {dt}s")