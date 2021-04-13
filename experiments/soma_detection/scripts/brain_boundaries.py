from brainlit.utils.session import NeuroglancerSession
import boto3
from tqdm import tqdm
import os
import numpy as np
import functools

brain_name = "brain1"

s3 = boto3.resource("s3")
bucket = s3.Bucket("open-neurodata")

brain_prefix = f"brainlit/{brain_name}"
segments_prefix = f"brainlit/{brain_name}_segments"
somas_prefix = f"brainlit/{brain_name}_somas"
skeletons_prefix = f"{segments_prefix}/skeletons"

brain_url = f"s3://open-neurodata/{brain_prefix}"
segments_url = f"s3://open-neurodata/{segments_prefix}"

for mip in [5, 6]:
    ngl_sess = NeuroglancerSession(mip=mip, url=brain_url, url_segments=segments_url)
    res = ngl_sess.cv_segments.scales[ngl_sess.mip]["resolution"]
    res_string = functools.reduce(lambda a, b: str(a) + "_" + str(b), res)

    bounds = [[np.Inf, -np.Inf], [np.Inf, -np.Inf], [np.Inf, -np.Inf]]

    voxels = bucket.objects.filter(Prefix=f"{brain_prefix}/{res_string}")
    for vox in tqdm(voxels):
        vox_id = os.path.basename(vox.key)
        if vox_id != "":
            coords = [[int(c) for c in dim.split("-")] for dim in vox_id.split("_")]

            for i, dim in enumerate(coords):
                bounds[i][0] = dim[0] if dim[0] < bounds[i][0] else bounds[i][0]
                bounds[i][1] = dim[1] if dim[1] > bounds[i][1] else bounds[i][1]

    min_bounds = np.array([dim[0] for dim in bounds])
    max_bounds = np.array([dim[1] for dim in bounds])

    min_bounds_spatial = np.multiply(min_bounds, res)
    max_bounds_spatial = np.multiply(max_bounds, res)
    print(min_bounds_spatial, max_bounds_spatial)
