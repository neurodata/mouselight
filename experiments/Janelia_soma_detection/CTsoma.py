from typing import Tuple
import numpy as np
from skimage import filters, morphology, measure
import pandas as pd
from scipy import ndimage
from brainlit.utils.util import check_type, check_iterable_type
import os
from brainlit.algorithms.detect_somas import find_somas
from pytest import raises
from pathlib import Path
from brainlit.utils.session import NeuroglancerSession
from cloudvolume.lib import Bbox
import glob
import napari

brain_url = "s3://open-neurodata/brainlit/brain1"
segments_url = "s3://open-neurodata/brainlit/brain1_segments"
ngl_sess = NeuroglancerSession(
    mip=2, url=brain_url, url_segments=segments_url, use_https=False
)
ImgPath = glob.glob(
    "experiments/soma_detection/data/brain1/volumes/*.npy"
)  # open directory folder to brainlit
NumSub = len(ImgPath)
for item in range(NumSub):
    volume_coords = (
        np.array(os.path.basename(ImgPath[item]).replace(".npy", "").split("_"))
        .astype(float)
        .round()
        .astype(int)
        .tolist()
    )
    res = ngl_sess.cv_segments.scales[ngl_sess.mip]["resolution"]
    _min = volume_coords[:3]
    _max = volume_coords[3:]
    vox_min = np.round(np.divide(_min, res)).astype(int)
    vox_max = np.round(np.divide(_max, res)).astype(int)
    bbox = Bbox(vox_min, vox_max)
    try:
        img = ngl_sess.pull_bounds_img(bounds=bbox)
        label, rel_centroids, out = find_somas(img, res)
        print("Item#", item, "label=", label, "rel_centroids=", rel_centroids)
        with napari.gui_qt():
            viewer = napari.view_image(
                img, name="Original Image", colormap="red", ndisplay=3
            )
            viewer.add_labels(
                out, name="find_soma" + str(item), num_colors=2, opacity=0.3
            )
            viewer.add_points(
                rel_centroids,
                size=3,
                symbol="o",
                face_color=np.array([0.67, 0.7, 1, 0.5]),
            )
    except:
        print("Disrupted at item", item)
