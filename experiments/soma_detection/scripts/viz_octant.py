import os
import boto3
from pathlib import Path
from cloudvolume.lib import Bbox
import numpy as np
from brainlit.utils.session import NeuroglancerSession
import math
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
import itertools

from skimage import (
    filters,
    morphology,
)

brain = 1
brain_name = f"brain{brain}"

s3 = boto3.resource("s3")
bucket = s3.Bucket("open-neurodata")

brain_prefix = f"brainlit/{brain_name}"
segments_prefix = f"brainlit/{brain_name}_segments"
octant_prefix = f"brainlit/{brain_name}_octant"
somas_prefix = f"brainlit/{brain_name}_somas"

brain_url = f"s3://open-neurodata/{brain_prefix}"
segments_url = f"s3://open-neurodata/{segments_prefix}"

ngl_sess = NeuroglancerSession(mip=4, url=brain_url, url_segments=segments_url)
ngl_sess2 = NeuroglancerSession(mip=1, url=brain_url, url_segments=segments_url)

res = ngl_sess.cv_segments.scales[ngl_sess.mip]["resolution"]
res2 = ngl_sess.cv_segments.scales[ngl_sess2.mip]["resolution"]

brain_lims = [10095672, 7793047, 13157636]
step = [100000, 100000, 100000]

origin = np.array([0, 0, 0])
center = np.array([math.ceil(lim / 2) for lim in brain_lims])

octants = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
octant_ids = [2, 0, 1, 3]

step = np.array([100000, 100000, 100000]).astype(float)
def discrete_to_spatial(discrete_x, discrete_y, discrete_z):
    discrete_coords = [discrete_x, discrete_y, discrete_z]
    spatial_coords = [[k * step[i] for i, k in enumerate(discrete_coords)], [(k + 1) * step[i] for i, k in enumerate(discrete_coords)]]
    return np.array(spatial_coords).astype(int)

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

fig = plt.figure(figsize=(5, 5))
axes = fig.subplots(2, 2)
for octant, octant_id in zip(octants, octant_ids):
    ax = axes[octant_id // 2, octant_id % 2]
    # ax.imshow(octant_proj, cmap="jet", origin="lower")
    
    octant_min = origin + np.multiply(octant, center)
    octant_max = center + np.multiply(octant, center)
        
    octant_min_vox = np.round(np.divide(octant_min, res)).astype(int)
    octant_max_vox = np.round(np.divide(octant_max, res)).astype(int)
        
    # octant_min_vox[-1] = 500
    # octant_max_vox[-1] = 550

    octant_bbox = Bbox(octant_min_vox, octant_max_vox)

    octant_volume = ngl_sess.pull_bounds_img(octant_bbox)

    octant_proj = np.amax(octant_volume, axis=2)
        
    ax.imshow(octant_proj, cmap="jet", origin="lower")
    ax.axis("off")
    
    if octant_id == 2:        
        print("Looking for somas in octant")
        for volume in bucket.objects.filter(Prefix=somas_prefix):
            volume_key = volume.key
            volume_id = os.path.basename(volume_key)
            if volume_id != "":
                volume_coords = np.array(volume_id.split("_")).astype(float)
                
                volume_min = volume_coords[:3]
                volume_max = volume_coords[3:]
                
                if np.all(volume_min > octant_min) and np.all(volume_max < octant_max):
                    print(f"{volume_id} is in octant")
                    
                    volume_vox_min = np.divide(volume_min, res2).astype(int)
                    volume_vox_max = np.divide(volume_max, res2).astype(int)
                    
                    volume_bbox = Bbox(volume_vox_min, volume_vox_max)
                    volume = ngl_sess2.pull_bounds_img(volume_bbox)
                    
                    try:
                        label, rel_somas, _ = contains_somas(volume)
                    except ValueError:
                        print("failed")
                    else:
                        if label == 1:
                            somas = np.array([np.multiply(volume_vox_min + c, res2) for c in rel_somas])
                            
                            for soma in somas:
                                soma_octant = np.divide(soma, res).astype(int)
                                ax.scatter(soma_octant[1], soma_octant[0], c="none", edgecolor="r", s=30, linewidth=2)
        
        # octant_somas_vox = []
        
        # region_vox_min = np.array([500, 1000, 500])
        # region_vox_max = np.array([600, 1250, 550])
        
        # region_min = np.multiply(region_vox_min, res).astype(float)
        # region_max = np.multiply(region_vox_max, res).astype(float)
        
        # region_min_id = np.divide(region_min, step).astype(int)
        # region_max_id = np.divide(region_max, step).astype(int)
        # print(region_min_id, region_max_id)

        # N = [range(_min, _max+1) for _min, _max in zip(region_min_id, region_max_id)]

        # _iter_discrete_coords = itertools.product(N[0], N[1], N[2])
        
        # for i, volume_coords in enumerate(itertools.starmap(discrete_to_spatial, _iter_discrete_coords)):
        #     volume_id = f"{volume_coords[0][0]}_{volume_coords[0][1]}_{volume_coords[0][2]}_{volume_coords[1][0]}_{volume_coords[1][1]}_{volume_coords[1][2]}"
        #     volume_key = f"{octant_prefix}/{volume_id}"
            
        #     bucket.download_file(volume_key, "tmp_volume.npy")
            
        #     volume_somas = np.load("tmp_volume.npy")
            
        #     print(volume_id)
        #     if len(volume_somas) > 0:
        #         print("hit")
        #         for volume_soma in volume_somas:
        #             volume_soma_vox = np.divide(volume_soma, res).astype(int)
        #             plt.scatter(volume_soma_vox[1], volume_soma_vox[0], color="r", marker="o", s=1, alpha=1)

        
        # volumes = bucket.objects.filter(Prefix=octant_prefix)
        # for volume in tqdm(volumes):
        #     volume_key = volume.key
        #     volume_id = os.path.basename(volume.key)
        #     if volume_id != "":
        #         bucket.download_file(volume_key, "tmp_volume.npy")
                
        #         volume_somas = np.load("tmp_volume.npy")
                
        #         if len(volume_somas) > 0:
        #             for volume_soma in volume_somas:
        #                 volume_soma_vox = np.divide(volume_soma, res).astype(int)
                        # if volume_soma_vox[0] > 100:
                        #     volume_coords = np.array(
                        #         volume_id.split("_")
                        #     ).astype(float)
                        #     volume_vox_min = np.round(np.divide(volume_coords[:3], res2)).astype(int)
                        #     volume_vox_max = np.round(np.divide(volume_coords[3:], res2)).astype(int)
                        #     print(volume_coords)

                
                        #     volume_img = ngl_sess2.pull_bounds_img(Bbox(volume_vox_min, volume_vox_max))
                        #     volume_proj = np.amax(volume_img, axis=2)
                        #     plt.imshow(volume_proj, origin="lower")
                        #     plt.savefig(f"test_volumes/volume_{len(octant_somas_vox)}.jpg")
                        #     print(volume_somas)
                        #     print(volume_soma_vox)
         #                octant_somas_vox.append(volume_soma_vox)
         #    if len(octant_somas_vox) == 100:
         #        break   

        # octant_somas_vox = np.array(octant_somas_vox)
        # plt.scatter(octant_somas_vox[:, 1], octant_somas_vox[:, 0], color="r", marker="o", s=10)

plt.savefig("figure.jpg")