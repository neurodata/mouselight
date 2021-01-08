# generate xml_import and terastitcher commands
from configparser import ConfigParser
import math
import argparse
from psutil import virtual_memory
import joblib
import boto3
import os
from util import (
    S3Url,
    upload_file_to_s3,
    download_file_from_s3,
    download_terastitcher_files,
    tqdm_joblib,
    aws_cli,
)
import subprocess
import shlex
from glob import glob
from tqdm import tqdm

parastitcher_path = f"{os.path.dirname(os.path.realpath(__file__))}/parastitcher.py"
paraconverter_path = f"{os.path.dirname(os.path.realpath(__file__))}/paraconverter.py"
python_path = os.path.expanduser("~/colm_pipeline_env/bin/python")

STITCH_ONLY, COMPUTE_ONLY, ALL_STEPS = range(3)


def write_import_xml(fname_importxml, scanned_matrix, metadata):
    img_regex = ".*.tiff"
    eofl = "\r\n"
    with open(fname_importxml, "w") as fp:
        fp.writelines(
            [
                f'<?xml version="1.0" encoding="UTF-8" ?>{eofl}',
                f'<!DOCTYPE TeraStitcher SYSTEM "TeraStitcher.DTD">{eofl}',
                f'<TeraStitcher volume_format="TiledXY|2Dseries">{eofl}',
                f"\t<stacks_dir value=\"{metadata['stack_dir']}\" />{eofl}",
                f'\t<ref_sys ref1="1" ref2="2" ref3="3" />{eofl}',
                f"\t<voxel_dims V=\"{metadata['voxel_size'][1]}\" H=\"{metadata['voxel_size'][0]}\" D=\"{metadata['voxel_size'][2]}\" />{eofl}",
                f"\t<origin V=\"{metadata['origin'][1]}\" H=\"{metadata['origin'][0]}\" D=\"{metadata['origin'][2]}\" />{eofl}",
                f"\t<mechanical_displacements V=\"{metadata['mechanical_displacements'][1]}\" H=\"{metadata['mechanical_displacements'][0]}\" />{eofl}",
                f"\t<dimensions stack_rows=\"{metadata['grid_size_Y']}\" stack_columns=\"{metadata['grid_size_X']}\" stack_slices=\"{metadata['num_slices']}\" />{eofl}",
                f"\t<STACKS>{eofl}",
            ]
        )
        # print(metadata['grid_size_Y'])
        # print(metadata['grid_size_X'])
        for j in range(metadata["grid_size_Y"]):
            for i in range(metadata["grid_size_X"]):
                abs_X_ef = i * metadata["abs_X"]
                abs_Y_ef = j * metadata["abs_Y"]
                folder_num = i + j * metadata["grid_size_X"]
                dir_name = f"LOC{folder_num:03}"
                if scanned_matrix[j][i] == "1":
                    loc_string = f"\t\t<Stack N_CHANS=\"1\" N_BYTESxCHAN=\"2\" ROW=\"{j}\" COL=\"{i}\" ABS_V=\"{abs_Y_ef}\" ABS_H=\"{abs_X_ef}\" ABS_D=\"0\" STITCHABLE=\"no\" DIR_NAME=\"{dir_name}\" Z_RANGES=\"[0,{metadata['num_slices']})\" IMG_REGEX=\"{img_regex}\">{eofl}"
                else:
                    loc_string = f'\t\t<Stack N_CHANS="1" N_BYTESxCHAN="2" ROW="{j}" COL="{i}" ABS_V="{abs_Y_ef}" ABS_H="{abs_X_ef}" ABS_D="0" STITCHABLE="no" DIR_NAME="" Z_RANGES="" IMG_REGEX="{img_regex}">{eofl}'
                fp.writelines(
                    [
                        loc_string,
                        f"\t\t\t<NORTH_displacements />{eofl}",
                        f"\t\t\t<EAST_displacements />{eofl}",
                        f"\t\t\t<SOUTH_displacements />{eofl}",
                        f"\t\t\t<WEST_displacements />{eofl}",
                        f"\t\t</Stack>{eofl}",
                    ]
                )
        fp.writelines([f"\t</STACKS>{eofl}", f"</TeraStitcher>{eofl}"])


def write_terastitcher_commands(fname_ts, metadata, stitched_dir, do_steps):
    eofl = "\n"
    subvoldim = 60
    # subvoldim = max(metadata['num_slices']//num_processes,20)
    mem = virtual_memory()
    num_cpus = joblib.cpu_count()
    num_processes = min(
        math.floor(
            mem.total
            / (
                (metadata["num_pix"] ** 2)
                * 4
                * (min(metadata["grid_size_X"], metadata["grid_size_Y"]) + 1)
                * subvoldim
            )
        )
        + 1,
        num_cpus,
    )
    depth = 5
    num_proc_merge = min(
        math.floor(mem.total / (metadata["height"] * metadata["width"] * 2 * depth)),
        num_cpus,
    )
    print(f"num processes to use for stitching is: {num_processes}")
    # step1 = f"terastitcher --test --projin={metadata['stack_dir']}/xml_import.xml --imout_depth=16 --sparse_data{eofl}"
    step2 = f"mpirun -n {num_processes} {python_path} {parastitcher_path} -2 --projin=\"{metadata['stack_dir']}/xml_import.xml\" --projout=\"{metadata['stack_dir']}/xml_displcomp.xml\" --sV={metadata['sV']} --sH={metadata['sH']} --sD={metadata['sD']} --subvoldim={subvoldim} --sparse_data --exectimes --exectimesfile=\"{metadata['stack_dir']}/t_displcomp\"{eofl}"
    step3 = f"terastitcher --displproj --projin=\"{metadata['stack_dir']}/xml_displcomp.xml\" --projout=\"{metadata['stack_dir']}/xml_displproj.xml\" --sparse_data{eofl}"
    step4 = f"terastitcher --displthres --projin=\"{metadata['stack_dir']}/xml_displproj.xml\" --projout=\"{metadata['stack_dir']}/xml_displthres.xml\" --threshold=0.3 --sparse_data{eofl}"
    step5 = f"terastitcher --placetiles --projin=\"{metadata['stack_dir']}/xml_displthres.xml\"{eofl}"
    step6 = f"mpirun -n {num_proc_merge} {python_path} {paraconverter_path} -s=\"{metadata['stack_dir']}/xml_merging.xml\" -d=\"{stitched_dir}\" --sfmt=\"TIFF (unstitched, 3D)\" --dfmt=\"TIFF (series, 2D)\" --height={metadata['height']} --width={metadata['width']} --depth={depth}{eofl}"
    ts_commands = []

    if do_steps == STITCH_ONLY:
        ts_commands.extend([step6])
    elif do_steps == COMPUTE_ONLY:
        ts_commands.extend([step2, step3, step4, step5])
    else:
        ts_commands.extend([step2, step3, step4, step5, step6])

    with open(fname_ts, "w") as fp:
        fp.writelines(ts_commands)

    return ts_commands


def get_metadata(path_to_config):
    metadata = {}

    config = ConfigParser()
    config.read(path_to_config)

    metadata["grid_size_X"] = int(
        config["North Scan Region"]["Num Horizontal"].strip('"')
    )
    metadata["grid_size_Y"] = int(
        config["North Scan Region"]["Num Vertical"].strip('"')
    )
    metadata["z_step"] = int(
        float(config["North Scan Region"]["Stack Step (mm)"].strip('"')) * 1000
    )

    metadata["num_slices"] = int(
        config["Experiment Settings"]["Num in stack (Top Left Corner)"].strip('"')
    )
    metadata["num_pix"] = int(config["Experiment Settings"]["X Resolution"].strip('"'))
    metadata["num_ch"] = int(
        config["Experiment Settings"]["Num Enabled Channels"].strip('"')
    )

    metadata["overlap_X"] = (
        float(
            config["North Scan Region Stats"]["Actual Horizontal Overlap (%)"].strip(
                '"'
            )
        )
        / 100
    )
    metadata["overlap_Y"] = (
        float(
            config["North Scan Region Stats"]["Actual Vertical Overlap (%)"].strip('"')
        )
        / 100
    )

    mag_idx = config["Objectives"]["North"].find("x") - 2
    metadata["mag"] = int(config["Objectives"]["North"][mag_idx : mag_idx + 2])

    metadata["num_pix"] = int(config["Experiment Settings"]["X Resolution"].strip('"'))
    metadata["num_ch"] = int(
        config["Experiment Settings"]["Num Enabled Channels"].strip('"')
    )
    metadata["scale_factor"] = 2048 / metadata["num_pix"]
    metadata["origin"] = (0, 0, 0)
    scale_factor = metadata["scale_factor"]
    if metadata["mag"] == 4:
        metadata["voxel_size"] = (
            1.46 * scale_factor,
            1.46 * scale_factor,
            metadata["z_step"],
        )
        # terastitcher parameters
        # X,Y,Z search radius in voxels to compute tile displacement
        metadata["sH"] = math.ceil(60 / scale_factor)
        metadata["sV"] = math.ceil(60 / scale_factor)
        metadata["sD"] = math.ceil(20 / scale_factor)

    elif metadata["mag"] == 10:
        metadata["voxel_size"] = (
            0.585 * scale_factor,
            0.585 * scale_factor,
            metadata["z_step"],
        )
        # terastitcher parameters
        # X,Y,Z search radius in voxels to compute tile displacement
        metadata["sH"] = 100
        metadata["sV"] = 60
        metadata["sD"] = 5
    elif metadata["mag"] == 25:
        metadata["voxel_size"] = (
            0.234 * scale_factor,
            0.234 * scale_factor,
            metadata["z_step"],
        )
        # terastitcher parameters
        # X,Y,Z search radius in voxels to compute tile displacement
        metadata["sH"] = math.ceil(60 / scale_factor)
        metadata["sV"] = math.ceil(60 / scale_factor)
        metadata["sD"] = math.ceil(20 / scale_factor)
    else:
        raise ("The only magnifications supported are 4,  10, or 25")
    metadata["mechanical_displacements"] = (
        math.floor(
            metadata["num_pix"]
            * (1 - metadata["overlap_X"])
            * metadata["voxel_size"][0]
        ),
        math.floor(
            metadata["num_pix"]
            * (1 - metadata["overlap_Y"])
            * metadata["voxel_size"][1]
        ),
    )
    metadata["abs_X"] = math.floor(metadata["num_pix"] * (1 - metadata["overlap_X"]))
    metadata["abs_Y"] = math.floor(metadata["num_pix"] * (1 - metadata["overlap_Y"]))
    metadata["width"] = math.ceil(
        metadata["abs_X"] * metadata["grid_size_X"]
        + metadata["num_pix"] * metadata["overlap_X"]
    )
    metadata["height"] = math.ceil(
        metadata["abs_Y"] * metadata["grid_size_Y"]
        + metadata["num_pix"] * metadata["overlap_Y"]
    )
    print(f"overlap_X: {metadata['overlap_X']}")
    print(f"overlap_Y: {metadata['overlap_Y']}")
    print(f"abs_X: {metadata['abs_X']}")
    print(f"abs_Y: {metadata['abs_Y']}")
    print(f"width: {metadata['width']}")
    print(f"height: {metadata['height']}")
    return metadata


def get_scanned_cells(fname_scanned_cells):
    # read scanned matrix file
    scanned_matrix = []
    with open(fname_scanned_cells, "r") as fp:
        for line in fp.readlines():
            x = line.strip().split(",")
            scanned_matrix.append(x)
    return scanned_matrix


def generate_stitching_commands(
    stitched_dir, stack_dir, metadata_s3_bucket, metadata_s3_path, do_steps=ALL_STEPS
):

    # download COLM metadata files
    scanned_cells_path = f"{stack_dir}/Scanned Cells.txt"
    config_file_path = f"{stack_dir}/Experiment.ini"
    s3 = boto3.resource("s3")
    s3.Object(
        metadata_s3_bucket, f"{metadata_s3_path}/Scanned Cells.txt"
    ).download_file(scanned_cells_path)
    s3.Object(metadata_s3_bucket, f"{metadata_s3_path}/Experiment.ini").download_file(
        config_file_path
    )

    # get metadata
    metadata = get_metadata(config_file_path)
    metadata["stack_dir"] = stack_dir

    # load scanned cells to indicate which locations contain data
    scanned_matrix = get_scanned_cells(scanned_cells_path)

    # write xml_import file for terastitcher
    fname_importxml = f"{stack_dir}/xml_import.xml"
    write_import_xml(fname_importxml, scanned_matrix, metadata)

    fname_ts = f"{stack_dir}/terastitcher_commands.sh"
    ts_commands = write_terastitcher_commands(
        fname_ts, metadata, stitched_dir, do_steps
    )

    return metadata, ts_commands


def run_terastitcher(
    raw_data_path,
    stitched_data_path,
    input_s3_path,
    log_s3_path=None,
    stitch_only=False,
    compute_only=False,
):

    input_s3_url = S3Url(input_s3_path.strip("/"))

    # generate commands to stitch data using Terastitcher
    # if stitch_only and not log_s3_path:
    #     raise("If using previous stitching results, must specify log_s3_path")
    # download terastitcher files if they arent already on local storage
    # download_terastitcher_files(log_s3_path, raw_data_path)

    if stitch_only:
        do_steps = STITCH_ONLY
    elif compute_only:
        do_steps = COMPUTE_ONLY
    else:
        do_steps = ALL_STEPS

    metadata, commands = generate_stitching_commands(
        stitched_data_path,
        raw_data_path,
        input_s3_url.bucket,
        input_s3_url.key,
        do_steps,
    )

    # run the Terastitcher commands
    for i in commands:
        print(i)
        subprocess.run(shlex.split(i))

    # # upload xml results to log_s3_path if not None
    # # and if not stitch_only
    if log_s3_path and not stitch_only:
        log_s3_url = S3Url(log_s3_path.strip("/"))
        files_to_save = glob(f"{raw_data_path}/*.xml")
        for i in tqdm(files_to_save, desc="saving xml files to S3"):
            out_path = i.split("/")[-1]
            upload_file_to_s3(i, log_s3_url.bucket, f"{log_s3_url.key}/{out_path}")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Create xml_import.xml file and terastitcher_commands.sh from Experiment.ini file"
    )
    parser.add_argument(
        "--stitched_dir",
        help="Directory to  store stitched tifs.",
        type=str,
        default="/home/ubuntu/ssd2/stitched_data",
    )
    parser.add_argument(
        "--stack_dir",
        help="Path to VW0 directory with tiles stored in LOC* folders.",
        type=str,
        default="/home/ubuntu/ssd1/VW0",
    )
    parser.add_argument(
        "--config_file",
        help="Path to Experiment.ini file",
        type=str,
        default="/home/ubuntu/ssd1/Experiment.ini",
    )
    parser.add_argument(
        "--scanned_cells",
        help="Path to Scanned Cells.txt file",
        type=str,
        default="/home/ubuntu/ssd1/Scanned Cells.txt",
    )
    parser.add_argument(
        "--stitch_only",
        help="If true, only run the stitching commands from existing xml_merging.xml file",
        type=bool,
        default=False,
    )

    args = parser.parse_args()

    generate_stitching_commands(
        args.stitched_dir,
        args.stack_dir,
        args.config_file,
        args.scanned_cells,
        args.stitch_only,
    )
