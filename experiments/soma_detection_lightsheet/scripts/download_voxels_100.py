from unicodedata import decimal
from cloudvolume.lib import Bbox
from napari import viewer
from brainlit.utils.session import NeuroglancerSession
import boto3
import os
from pathlib import Path
import numpy as np
import time
from tqdm import tqdm
import napari

upload = False
viz = False

cwd = Path(os.path.abspath(__file__))
exp_dir = cwd.parents[1]
data_dir = os.path.join(exp_dir, "data")

if os.path.exists(data_dir) is False:
    os.makedirs(data_dir)

brains = [1, 2]
s3 = boto3.resource("s3")
bucket = s3.Bucket("open-neurodata")

for brain in brains[:1]:
    brain_name = "brain%d" % brain

    brain_dir = os.path.join(data_dir, brain_name)
    volumes_dir = os.path.join(brain_dir, "volumes")
    if os.path.exists(volumes_dir) is False:
        os.makedirs(volumes_dir)

    brain_prefix = f"brainlit/{brain_name}"
    segments_prefix = f"brainlit/{brain_name}_segments"
    somas_prefix = f"brainlit/{brain_name}_somas"
    skeletons_prefix = f"{segments_prefix}/skeletons"

    brain_url = f"s3://open-neurodata/{brain_prefix}"
    segments_url = f"s3://open-neurodata/{segments_prefix}"

    ngl_sess = NeuroglancerSession(mip=1, url=brain_url, url_segments=segments_url)

    segments = bucket.objects.filter(Prefix=skeletons_prefix)
    n = sum(1 if os.path.basename(seg.key) != "info" else 0 for seg in segments.all())
    vertices_filename = os.path.join(brain_dir, "vertices.npy")
    if os.path.exists(vertices_filename):
        print(f"Found existing somas saved for brain {brain}")
        vertex_coords = np.load(vertices_filename, allow_pickle=True)
    else:
        print(f"Downloading somas for brain {brain}")
        vertex_coords = np.zeros((n, 3))
        i = 0
        for seg in tqdm(segments):
            seg_id = os.path.basename(seg.key)
            if seg_id != "info":
                vertices = ngl_sess.cv_segments.skeleton.get(int(seg_id)).vertices
                # vertices coordinates are in nm
                soma_vertex = vertices[0]
                vertex_coords[i] = soma_vertex
                i += 1
        np.save(vertices_filename, vertex_coords, allow_pickle=True)

    # find closest soma vertex to origin to start pulling volumes
    vertex_norms = [np.linalg.norm(v) for v in vertex_coords]
    sorted_coords = vertex_coords[np.argsort(vertex_norms)]
    sorted_norms = [np.linalg.norm(v) for v in sorted_coords]

    # download volumes
    size = int(100e3)  # 100 x 10^3 nm = 100 um
    while len(sorted_coords) > 0:
        # count somas that are left
        print(f"============\nThere are {len(sorted_coords)}/{n} somas left to pull")
        # get coordinates of soma in nm
        v = sorted_coords[0]
        # get volume bbox coordinates in nm
        _min = v - size / 2 * np.ones(3)
        _max = v + size / 2 * np.ones(3)
        # check if other somas fall into the same volume
        contained = np.array(
            [(v > _min).all() and (v < _max).all() for v in sorted_coords]
        )
        assert contained[0] == True
        contained_ids = np.where(contained == True)[0]
        print(f"There are {len(contained_ids)} somas contained in {_min}{_max}")
        contained_coords = sorted_coords[contained_ids]
        # convert spatial coords to voxel coords
        res = ngl_sess.cv_segments.scales[ngl_sess.mip]["resolution"]
        vox_min = np.round(np.divide(_min, res)).astype(int)
        vox_max = np.round(np.divide(_max, res)).astype(int)
        bbox = Bbox(vox_min, vox_max)
        bbox_list = bbox.to_list()
        print(
            f"Pulling volume, bbox = {bbox}...",
            end="",
            flush=True,
        )
        t0 = time.time()
        volume = ngl_sess.pull_bounds_img(bbox)
        t = time.time()
        dt = np.around(t - t0, decimals=3)
        print(f"done in {dt}s")
        # compute relative voxel coordinates of somas
        rel_vox_c = np.array(
            [
                np.round(np.divide(c, res)).astype(int) - vox_min
                for c in contained_coords
            ]
        )
        # visualize volumes with points on top of somas
        if viz:
            with napari.gui_qt():
                viewer = napari.Viewer(ndisplay=3)
                viewer.add_image(volume)
                viewer.add_points(
                    rel_vox_c,
                    size=5,
                    symbol="o",
                    face_color=np.array([1, 0, 0, 0.5]),
                )
        # save coords
        coords_filename = f"{_min[0]}_{_min[1]}_{_min[2]}_{_max[0]}_{_max[1]}_{_max[2]}"
        coords_path = os.path.join(volumes_dir, f"{coords_filename}.npy")

        np.save(coords_path, contained_coords, allow_pickle=True)

        # upload coords to s3
        if upload:
            coords_s3key = f"{somas_prefix}/{coords_filename}"
            print(f"Uploading coordinates to S3...", end="", flush=True)
            t0 = time.time()
            bucket.upload_file(coords_path, coords_s3key)
            t = time.time()
            dt = np.around(t - t0, decimals=3)
            print(f"done in {dt}s")
        # update spatial_coords
        sorted_coords = np.delete(sorted_coords, contained_ids, axis=0)
