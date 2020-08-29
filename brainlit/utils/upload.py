import math
from cloudvolume import CloudVolume, Skeleton, storage
from cloudvolume.frontends.precomputed import CloudVolumePrecomputed
import numpy as np
import joblib
from joblib import Parallel, delayed, cpu_count
import joblib
from glob import glob
import argparse
from psutil import virtual_memory
from typing import Optional, Sequence, Union, Literal, Tuple, List
import contextlib

import tifffile as tf
from pathlib import Path
from .swc import swc2skeleton
import time
from tqdm.auto import tqdm
from .util import (
    tqdm_joblib,
    check_type,
    check_iterable_type,
    check_size,
    check_precomputed,
    check_binary_path,
)


def get_volume_info(
    image_dir: str,
    num_resolutions: int,
    channel: Optional[int] = 0,
    extension: Optional[Literal["tif"]] = "tif",
) -> Tuple[List, List, List, Tuple, List]:
    """Get filepaths along the octree-format image directory

    Arguments:
        image_dir: Filepath to HIGHEST LEVEL(lowest res) of octree dir.
        num_resolutions: Number of resolutions for which downsampling has been done.
        channel: Channel number to upload.
        extension: File extension of image files.
    Returns:
        files_ordered: List of file paths, 1st dim contains list for each res.
        paths_bin: List of binary paths, 1st dim contains lists for each res.
        vox_size: List of highest resolution voxel sizes (nm).
        tiff_dims: (x,y,z) voxel dimensions for a single tiff image.
    """
    check_type(image_dir, str)
    check_type(num_resolutions, int)
    check_type(channel, int)
    check_type(extension, str)
    if num_resolutions < 1:
        raise ValueError(f"Number of resolutions should be > 0, not {num_resolutions}")
    check_type(channel, int)
    if channel < 0:
        raise ValueError(f"Channel should be >= 0, not {channel}")
    check_type(extension, str)
    if extension not in ["tif"]:
        raise ValueError(f"{extension} should be 'tif'")

    def RepresentsInt(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    p = Path(image_dir)
    files = [i.parts for i in p.rglob(f"*.{channel}.{extension}")]
    parent_dirs = len(p.parts)

    files_ordered = [
        [i for i in files if len(i) == j + parent_dirs + 1]
        for j in range(num_resolutions)
    ]
    paths_bin = [
        [[f"{int(j)-1:03b}" for j in k if len(j) == 1 and RepresentsInt(j)] for k in i]
        for i in files_ordered
    ]
    for i, resolution in enumerate(files_ordered):
        for j, filepath in enumerate(resolution):
            files_ordered[i][j] = str(Path(*filepath))

    img_size = np.squeeze(tf.imread(str(p / "default.0.tif"))).T.shape
    transform = open(str(p / "transform.txt"), "r")
    vox_size = [
        float(s[4:].rstrip("\n")) * (0.5 ** (num_resolutions - 1))
        for s in transform.readlines()
        if "s" in s
    ]
    transform = open(str(p / "transform.txt"), "r")
    origin = [int(o[4:].rstrip("\n")) / 1000 for o in transform.readlines() if "o" in o]
    return files_ordered, paths_bin, vox_size, img_size, origin


def create_cloud_volume(
    precomputed_path: str,
    img_size: Sequence[int],
    voxel_size: Sequence[Union[int, float]],
    num_resolutions: int,
    chunk_size: Optional[Sequence[int]] = None,
    parallel: Optional[bool] = False,
    layer_type: Optional[Literal["image", "segmentaion", "annotation"]] = "image",
    dtype: Optional[Literal["uint16", "uint64"]] = None,
    commit_info: Optional[bool] = True,
) -> CloudVolumePrecomputed:
    """Create CloudVolume object and info file.

    Handles both image volumes and segmentation volumes from octree structure.

    Arguments:
        precomputed_path: cloudvolume path
        img_size: x, y, z voxel dimensions of tiff images.
        voxel_size: x, y, z dimensions of highest res voxel size (nm).
        num_resolutions: The number of resolutions to upload.
        chunk_size: The size of chunks to use for upload. If None, uses img_size/2.
        parallel: Whether to upload chunks in parallel.
        layer_type: The type of cloudvolume object to create.
        dtype: The data type of the volume. If None, uses default for layer type.
        commit_info: Whether to create an info file at the path, defaults to True.
    Returns:
        vol: Volume designated for upload.
    """
    # defaults
    if chunk_size is None:
        chunk_size = [int(i / 4) for i in img_size]  # /2 took 42 hrs
    if dtype is None:
        if layer_type == "image":
            dtype = "uint16"
        elif layer_type == "segmentation" or layer_type == "annotation":
            dtype = "uint64"
        else:
            raise ValueError(
                f"layer type is {layer_type}, when it should be image or str"
            )

    # check inputs
    check_precomputed(precomputed_path)
    check_size(img_size, allow_float=False)
    check_size(voxel_size)
    check_type(num_resolutions, int)
    if num_resolutions < 1:
        raise ValueError(f"Number of resolutions should be > 0, not {num_resolutions}")
    check_size(chunk_size)
    check_type(parallel, bool)
    check_type(layer_type, str)
    if layer_type not in ["image", "segmentation", "annotation"]:
        raise ValueError(
            f"{layer_type} should be 'image', 'segmentation', or 'annotation'"
        )
    check_type(dtype, str)
    if dtype not in ["uint16", "uint64"]:
        raise ValueError(f"{dtype} should be 'uint16' or 'uint64'")
    check_type(commit_info, bool)

    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type=layer_type,
        data_type=dtype,  # Channel images might be 'uint8'
        encoding="raw",  # raw, jpeg, compressed_segmentation, fpzip, kempressed
        resolution=voxel_size,  # Voxel scaling, units are in nanometers
        voxel_offset=[0, 0, 0],  # x,y,z offset in voxels from the origin
        chunk_size=chunk_size,  # units are voxels
        volume_size=[i * 2 ** (num_resolutions - 1) for i in img_size],
    )
    vol = CloudVolume(precomputed_path, info=info, parallel=parallel)
    [
        vol.add_scale((2 ** i, 2 ** i, 2 ** i), chunk_size=chunk_size)
        for i in range(num_resolutions)
    ]
    if commit_info:
        vol.commit_info()
    if layer_type == "image" or layer_type == "annotation":
        vols = [
            CloudVolume(precomputed_path, mip=i, parallel=parallel)
            for i in range(num_resolutions - 1, -1, -1)
        ]
    elif layer_type == "segmentation":
        info.update(skeletons="skeletons")

        skel_info = {
            "@type": "neuroglancer_skeletons",
            "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            "vertex_attributes": [
                {"id": "radius", "data_type": "float32", "num_components": 1},
                {"id": "vertex_types", "data_type": "float32", "num_components": 1},
                {"id": "vertex_color", "data_type": "float32", "num_components": 4},
            ],
        }
        with storage.SimpleStorage(vol.cloudpath) as stor:
            stor.put_json(str(Path("skeletons") / "info"), skel_info)
        vols = [vol]
    return vols


def get_data_ranges(
    bin_path: List[List[str]], chunk_size: Tuple[int, int, int]
) -> Tuple[List[int], List[int], List[int]]:
    """Get ranges (x,y,z) for chunks to be stitched together in volume

    Arguments:
        bin_path: Binary paths to files.
        chunk_size: The size of chunk to get ranges over.
    Returns:
        x_range: x-coord int bounds.
        y_range: y-coord int bounds.
        z_range: z-coord int bounds.
    """
    for b in bin_path:
        check_binary_path(b)
    check_size(chunk_size)

    x_curr, y_curr, z_curr = 0, 0, 0
    tree_level = len(bin_path)
    print(bin_path)
    for idx, i in enumerate(bin_path):
        print(i)
        scale_factor = 2 ** (tree_level - idx - 1)
        x_curr += int(i[2]) * chunk_size[0] * scale_factor
        y_curr += int(i[1]) * chunk_size[1] * scale_factor
        # flip z axis so chunks go anterior to posterior
        z_curr += int(i[0]) * chunk_size[2] * scale_factor
    x_range = [x_curr, x_curr + chunk_size[0]]
    y_range = [y_curr, y_curr + chunk_size[1]]
    z_range = [z_curr, z_curr + chunk_size[2]]
    return x_range, y_range, z_range


def process(file_path: str, bin_path: List[str], vol: CloudVolumePrecomputed):
    """The parallelizable method to upload data.

    Loads the image into memory, and pushes it to specific ranges in the CloudVolume.

    Arguments:
        file_path: Path to the image file.
        bin_path: Binary path to the image file.
        vol: CloudVolume object to upload.
    """
    check_type(file_path, str)
    check_binary_path(bin_path)
    check_type(vol, CloudVolumePrecomputed)

    array = tf.imread(file_path).T
    ranges = get_data_ranges(bin_path, vol.scales[-1]["size"])
    vol[
        ranges[0][0] : ranges[0][1],
        ranges[1][0] : ranges[1][1],
        ranges[2][0] : ranges[2][1],
    ] = array
    return


def upload_volumes(
    input_path: str,
    precomputed_path: str,
    num_mips: int,
    parallel: bool = False,
    chosen: int = -1,
):
    """Uploads image data from local to a precomputed path.

    Specify num_mips for additional resolutions. If `chosen` is used, an info file will not be generated.

    Arguments:
        input_path: The filepath to the root directory of the octree image data.
        precomputed_path: CloudVolume precomputed path or url.
        num_mips: The number of resolutions to upload.
        parallel: Whether to upload in parallel. Default is False.
        chosen: If not -1, uploads only that specific mip. Default is -1.
    """
    check_type(input_path, str)
    check_precomputed(precomputed_path)
    check_type(num_mips, int)
    if num_mips < 1:
        raise ValueError(f"Number of resolutions should be > 0, not {num_mips}")
    check_type(parallel, bool)
    check_type(chosen, int)
    if chosen < -1 or chosen >= num_mips:
        raise ValueError(f"{chosen} should be -1, or between 0 and {num_mips-1}")

    (files_ordered, paths_bin, vox_size, img_size, _) = get_volume_info(
        input_path, num_mips,
    )
    if chosen != -1:
        commit_info = False
    else:
        commit_info = True
    vols = create_cloud_volume(
        precomputed_path,
        img_size,
        vox_size,
        num_mips,
        parallel=parallel,
        layer_type="image",
        commit_info=commit_info,
    )

    num_procs = min(
        math.floor(
            virtual_memory().total / (img_size[0] * img_size[1] * img_size[2] * 8)
        ),
        cpu_count(),
    )
    start = time.time()
    if chosen == -1:
        for mip, vol in enumerate(vols):
            try:
                with tqdm_joblib(
                    tqdm(
                        desc="Creating precomputed volume",
                        total=len(files_ordered[mip]),
                    )
                ) as progress_bar:
                    Parallel(num_procs, timeout=1800)(
                        delayed(process)(f, b, vols[mip],)
                        for f, b in zip(files_ordered[mip], paths_bin[mip])
                    )
                print(f"\nFinished mip {mip}, took {time.time()-start} seconds")
                start = time.time()
            except Exception as e:
                print(e)
                print("timed out on a slice. moving on to the next step of pipeline")
    else:
        try:
            with tqdm_joblib(
                tqdm(
                    desc="Creating precomputed volume", total=len(files_ordered[chosen])
                )
            ) as progress_bar:
                Parallel(num_procs, timeout=1800)(
                    delayed(process)(f, b, vols[chosen],)
                    for f, b in zip(files_ordered[chosen], paths_bin[chosen])
                )
            print(f"\nFinished mip {chosen}, took {time.time()-start} seconds")
        except Exception as e:
            print(e)
            print("timed out on a slice. moving on to the next step of pipeline")


def create_skel_segids(
    swc_dir: str, origin: Sequence[Union[int, float]]
) -> Tuple[Skeleton, List[int]]:
    """ Create skeletons to be uploaded as precomputed format

    Arguments:
        swc_dir: Path to consensus swc files.
        origin: x,y,z coordinate of coordinate frame in space in mircons.

    Returns:
        skeletons: .swc skeletons to be pushed to bucket.
        segids: List of ints for each swc's label.
    """
    check_type(swc_dir, str)
    check_size(origin)

    p = Path(swc_dir)
    files = [str(i) for i in p.glob("*.swc")]
    if len(files) == 0:
        raise FileNotFoundError(f"No .swc files found in {swc_dir}.")
    skeletons = []
    segids = []
    for i in tqdm(files, desc="converting swcs to neuroglancer format..."):
        skeletons.append(swc2skeleton(i, origin=origin))
        segids.append(skeletons[-1].id)
    return skeletons, segids


def upload_segments(input_path, precomputed_path, num_mips):
    """Uploads segmentation data from local to precomputed path.

    Arguments:
        input_path: The filepath to the root directory of the octree data with consensus-swcs folder.
        precomputed_path: CloudVolume precomputed path or url.
        num_mips: The number of resolutions to upload (for info file).
    """
    check_type(input_path, str)
    check_precomputed(precomputed_path)
    check_type(num_mips, int)
    if num_mips < 1:
        raise ValueError(f"Number of resolutions should be > 0, not {num_mips}")

    (_, _, vox_size, img_size, origin) = get_volume_info(input_path, num_mips,)
    print(vox_size)
    vols = create_cloud_volume(
        precomputed_path, img_size, vox_size, num_mips, layer_type="segmentation",
    )
    swc_dir = Path(input_path) / "consensus-swcs"
    segments, segids = create_skel_segids(str(swc_dir), origin)
    for skel in segments:
        vols[0].skeleton.upload(skel)


def upload_annotations(input_path, precomputed_path, num_mips):
    """Uploads empty annotation volume.
    """
    (_, _, vox_size, img_size, origin) = get_volume_info(input_path, num_mips,)
    create_cloud_volume(
        precomputed_path, img_size, vox_size, num_mips, layer_type="annotation",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert local volume into precomputed volume on S3."
    )
    parser.add_argument(
        "input_path",
        help="Path to directory containing stitched tiles named sequentially.",
    )
    parser.add_argument(
        "precomputed_path",
        help="Path to location where precomputed volume should be stored. Example: s3://<bucket>/<experiment>/<channel>",
    )
    parser.add_argument(
        "layer_type", help="Layer type to upload. One of ['image', 'segmentation']",
    )
    parser.add_argument(
        "--extension",
        help="Extension of stitched files. default is tif",
        default="tif",
        type=str,
    )
    parser.add_argument(
        "--channel", help="Channel to upload to", default=0, type=int,
    )
    parser.add_argument(
        "--num_resolutions",
        help="Number of resoltions for  which downsampling has been done. Default: 7",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--chosen_res",
        help="Specified resolution to upload. 0 is highest. Default uploads all",
        default=-1,
        type=int,
    )

    args = parser.parse_args()
    if args.layer_type == "image":
        upload_volumes(
            args.input_path,
            args.precomputed_path,
            args.num_resolutions,
            chosen=args.chosen_res,
        )
    elif args.layer_type == "segmentation":
        upload_segments(
            args.input_path, args.precomputed_path, args.num_resolutions,
        )
    else:
        upload_segments(
            args.input_path, args.precomputed_path + "_segments", args.num_resolutions,
        )
        upload_volumes(
            args.input_path,
            args.precomputed_path,
            args.num_resolutions,
            chosen=args.chosen_res,
        )
