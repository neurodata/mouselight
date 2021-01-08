import time
from tqdm import tqdm
import tifffile as tf
import numpy as np
import argparse
import os
import SimpleITK as sitk
import math
from cloudvolume import CloudVolume
import tinybrain
from psutil import virtual_memory
import joblib
from joblib import Parallel, delayed

# from concurrent.futures import ProcessPoolExecutor


def create_cloud_volume(
    precomputed_path,
    img_size,
    voxel_size,
    dtype="uint16",
    num_hierarchy_levels=5,
    parallel=True,
):
    if dtype == "uint64":
        layer_type = "segmentation"
    else:
        layer_type = "image"
    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type=layer_type,
        data_type=dtype,  # Channel images might be 'uint8'
        encoding="raw",  # raw, jpeg, compressed_segmentation, fpzip, kempressed
        resolution=voxel_size,  # Voxel scaling, units are in nanometers
        voxel_offset=[0, 0, 0],  # x,y,z offset in voxels from the origin
        # Pick a convenient size for your underlying chunk representation
        # Powers of two are recommended, doesn't need to cover image exactly
        chunk_size=[512, 512, 1],  # units are voxels
        volume_size=img_size,  # e.g. a cubic millimeter dataset
    )
    vol = CloudVolume(precomputed_path, info=info, parallel=parallel)
    # add mip 1
    [
        vol.add_scale((2 ** i, 2 ** i, 1), chunk_size=[512, 512, 1])
        for i in range(num_hierarchy_levels)
    ]
    vol.commit_info()
    return vol


def process(z, img):
    global layer_path, num_mips
    vols = [CloudVolume(layer_path, i, parallel=False) for i in range(num_mips)]
    if img.dtype in (np.uint8, np.uint16, np.float32, np.float64):
        img_pyramid = tinybrain.accelerated.average_pooling_2x2(img, num_mips=num_mips)
    else:
        img_pyramid = tinybrain.accelerated.mode_pooling_2x2(img, num_mips=num_mips)
    vols[0][:, :, z] = img[:, :, None]
    for i in range(num_mips - 1):
        vols[i + 1][:, :, z] = img_pyramid[i][:, :, None]


def ingest_image_stack(s3_path, voxel_size, img_stack, extension, dtype):

    if extension == "tif":
        img = tf.imread(os.path.expanduser(img_stack))
    else:
        tmp = sitk.ReadImage(os.path.expanduser(img_stack))
        img = sitk.GetArrayFromImage(tmp)
    img = np.asarray(img, dtype=dtype)

    img_size = img.shape[::-1]
    vol = create_cloud_volume(s3_path, img_size, voxel_size, dtype=dtype)

    mem = virtual_memory()
    num_procs = min(
        math.floor(mem.total / (img.shape[0] * img.shape[1] * 8)), joblib.cpu_count()
    )
    print(f"num processes: {num_procs}")
    print(f"layer path: {vol.layer_cloudpath}")
    global layer_path, num_mips
    num_mips = 3
    layer_path = vol.layer_cloudpath

    data = [(i, img.T[:, :, i]) for i in range(img.shape[0])]
    files = [i[1] for i in data]
    zs = [i[0] for i in data]

    Parallel(num_procs)(
        delayed(process)(z, f) for z, f in tqdm(zip(zs, files), total=len(zs))
    )
    # with ProcessPoolExecutor(max_workers=num_procs) as executor:
    #     executor.map(process, zs, files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest an image stack into S3.")
    parser.add_argument(
        "-s3_path", help="S3 path to store image as precomputed volume. ", type=str
    )
    parser.add_argument(
        "-img_stack", help="Path to image stack to be uploaded", type=str
    )
    parser.add_argument("-fmt", help="extension of file. can be tif or img", type=str)
    parser.add_argument(
        "-voxel_size",
        help="Voxel size of image. 3 numbers in nanometers",
        nargs=3,
        type=float,
    )
    parser.add_argument("--dtype", help="Datatype of image", type=str, default="uint16")

    args = parser.parse_args()

    ingest_image_stack(
        args.s3_path, args.voxel_size, args.img_stack, args.fmt, args.dtype
    )
