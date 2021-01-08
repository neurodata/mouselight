import shlex
from download_data import download_data
from util import S3Url
from cloudvolume import CloudVolume
from scipy.spatial.transform import Rotation
import numpy as np
from util import get_reorientations, aws_cli
from visualization import (
    ara_average_data_link,
    ara_annotation_data_link,
    create_viz_link,
)
import argparse
import subprocess
import os
from ingest_image_stack import ingest_image_stack

atlas_orientation = "PIR"


def get_affine_matrix(
    translation,
    rotation,
    from_orientation,
    to_orientation,
    fixed_scale,
    s3_path,
    center=False,
):
    """
    translation: x,y,z  translations respectively in microns
    center: if not None, should be an S3 path to compute the origin from
    """
    # since neuroglancer uses corner 0 coordinates we need to center the volume at it's center
    vol = CloudVolume(s3_path)
    # volume size in um
    vol_size = np.multiply(vol.scales[0]["size"], vol.scales[0]["resolution"]) / 1e3
    # make affine matrix in homogenous coordinates
    affine = np.zeros((4, 4))
    affine[-1, -1] = 1
    order, flips = get_reorientations(from_orientation, to_orientation)
    # reorder vol_size to match reorientation
    vol_size = vol_size[order]
    dim = affine.shape[0]
    # swap atlas axes to match target
    affine[range(len(order)), order] = 1
    # flip across appropriate dimensions
    affine[:3, :3] = np.diag(flips) @ affine[:3, :3]

    if center:
        # for each flip add the size of image in that dimension
        affine[:3, -1] += np.array(
            [vol_size[i] if flips[i] == -1 else 0 for i in range(len(flips))]
        )
        # make image centered at the middle of the image
        # volume is now centered
        affine[:3, -1] -= vol_size / 2

    # get rotation matrix
    if np.array(rotation).any():
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = Rotation.from_euler(
            "xyz", rotation, degrees=True
        ).as_matrix()
        # compose rotation with affine
        affine = rotation_matrix @ affine
    # add translation components
    # note: for neuroglancer affine, we scale the translations by voxel size
    # because neuroglancer expects translation in voxels
    affine[:3, -1] += translation

    # scale by fixed_scale
    affine = np.diag([fixed_scale, fixed_scale, fixed_scale, 1.0]) @ affine

    return affine


def register(
    input_s3_path,
    output_s3_path,
    log_s3_path,
    orientation,
    fixed_scale,
    translation,
    rotation,
    missing_data_correction,
    grid_correction,
    bias_correction,
    regularization,
    num_iterations,
):

    # registration
    # get channel name
    print(input_s3_path)
    s3_url = S3Url(input_s3_path)
    channel = s3_url.key.split("/")[-1]
    exp = s3_url.key.split("/")[-2]

    # only after stitching autofluorescence channel
    base_path = os.path.expanduser("~/")
    registration_prefix = f"{base_path}/{exp}_{channel}_registration/"
    target_name = f"{base_path}/autofluorescence_data.tif"

    # download downsampled autofluorescence channel
    print("downloading data for registration...")
    voxel_size = download_data(input_s3_path, target_name)
    # if high res atlas labels file doesn't exist
    ara_annotation_10um = os.path.expanduser(
        "~/CloudReg/registration/atlases/ara_annotation_10um.tif"
    )
    if not os.path.exists(ara_annotation_10um):
        # download it
        _ = download_data(
            ara_annotation_data_link(10), ara_annotation_10um, desired_resolution=10000
        )

    # initialize affine transformation for data
    atlas_res = 100
    atlas_s3_path = ara_average_data_link(atlas_res)
    initial_affine = get_affine_matrix(
        translation,
        rotation,
        atlas_orientation,
        orientation,
        fixed_scale,
        atlas_s3_path,
    )

    # run registration
    affine_string = [", ".join(map(str, i)) for i in initial_affine]
    affine_string = "; ".join(affine_string)
    matlab_registration_command = f"""
        matlab -nodisplay -nosplash -nodesktop -r \"niter={num_iterations};sigmaR={regularization};missing_data_correction={int(missing_data_correction)};grid_correction={int(grid_correction)};bias_correction={int(bias_correction)};base_path=\'{base_path}\';target_name=\'{target_name}\';registration_prefix=\'{registration_prefix}\';dxJ0={voxel_size};fixed_scale={fixed_scale};initial_affine=[{affine_string}];run(\'~/CloudReg/registration/registration_script_mouse_GN.m\')\"
    """
    print(matlab_registration_command)
    subprocess.run(shlex.split(matlab_registration_command))

    # save results to S3
    if log_s3_path:
        # sync registration results to log_s3_path
        aws_cli(["s3", "sync", registration_prefix, log_s3_path])

    # upload high res deformed atlas and deformed target to S3
    ingest_image_stack(
        output_s3_path,
        voxel_size,
        f"{registration_prefix}/downloop_2_labels_to_target_highres.img",
        "img",
        "uint64",
    )

    # print out viz link for visualization
    # visualize results at 5 microns
    viz_link = create_viz_link(
        [input_s3_path, output_s3_path], output_resolution=np.array([5] * 3) / 1e6
    )
    print("###################")
    print(f"VIZ LINK: {viz_link}")
    print("###################")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Run COLM pipeline on remote EC2 instance with given input parameters"
    )
    # data args
    parser.add_argument(
        "-input_s3_path",
        help="S3 path to precomputed volume used to register the data",
        type=str,
    )
    parser.add_argument(
        "-log_s3_path",
        help="S3 path at which registration outputs are stored.",
        type=str,
    )
    parser.add_argument(
        "--output_s3_path",
        help="S3 path to store atlas transformed to target as precomputed volume. Should be of the form s3://<bucket>/<path_to_precomputed>. Default is same as input s3_path with atlas_to_target as channel name",
        type=str,
        default=None,
    )

    # affine initialization args
    parser.add_argument(
        "-orientation", help="3-letter orientation of data. i.e. LPS", type=str
    )
    parser.add_argument(
        "--scale",
        help="Fixed scale of data, uniform in all dimensions. Default is 1.",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--translation",
        help="Initial translation in x,y,z respectively in microns.",
        nargs="+",
        type=float,
        default=[0, 0, 0],
    )
    parser.add_argument(
        "--rotation",
        help="Initial rotation in x,y,z respectively in degrees.",
        nargs="+",
        type=float,
        default=[0, 0, 0],
    )

    # preprocessing args
    parser.add_argument(
        "--bias_correction",
        help="Perform bias correction prior to registration.",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--missing_data_correction",
        help="Perform missing data correction by ignoring 0 values in image prior to registration.",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--grid_correction",
        help="Perform correction for low-intensity grid artifact (COLM data)",
        type=bool,
        default=True,
    )

    # registration params
    parser.add_argument(
        "--regularization",
        help="Weight of the regularization. Bigger regularization means less regularization. Default is 5e3",
        type=float,
        default=5e3,
    )
    parser.add_argument(
        "--iterations",
        help="Number of iterations to do at low resolution. Default is 5000.",
        type=int,
        default=5000,
    )

    args = parser.parse_args()

    register(
        args.input_s3_path,
        args.output_s3_path,
        args.log_s3_path,
        args.orientation,
        args.scale,
        args.translation,
        args.rotation,
        args.missing_data_correction,
        args.grid_correction,
        args.bias_correction,
        args.regularization,
        args.iterations,
    )
