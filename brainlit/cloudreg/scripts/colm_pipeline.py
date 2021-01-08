from download_raw_data import download_raw_data
from correct_raw_data import correct_raw_data
from create_precomputed_volume import create_precomputed_volume
from correct_stitched_data import correct_stitched_data
from stitching import run_terastitcher
from util import (
    S3Url,
    upload_file_to_s3,
    download_file_from_s3,
    download_terastitcher_files,
    tqdm_joblib,
    aws_cli,
)
from visualization import create_viz_link
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse
import os
from joblib import Parallel, delayed
import shutil


def colm_pipeline(
    input_s3_path,
    output_s3_path,
    channel_of_interest,
    autofluorescence_channel,
    raw_data_path,
    stitched_data_path,
    log_s3_path=None,
):
    """
    input_s3_path: S3 path to raw COLM data. Should be of the form s3://<bucket>/<experiment>
    output_s3_path: S3 path to store precomputed volume. Precomputed volumes for each channel will be stored under this path. Should be of the form s3://<bucket>/<path_to_precomputed>
    channel_of_interest: Channel number to operate on. Should be a single integer.
    autofluorescence_channel: Autofluorescence channel number. Should be a single integer.
    raw_data_path: Local path where corrected raw data will be stored.
    stitched_data_path: Local path where stitched slices will be stored.
    log_s3_path: S3 path at which pipeline intermediates can be stored including bias correction tile and xml files from Terastitcher.

    """
    # get the metadata file paths specific for COLM
    input_s3_url = S3Url(input_s3_path.strip("/"))
    output_s3_url = S3Url(output_s3_path.strip("/"))

    # download raw data onto local SSD
    vw0_path = f"{input_s3_url.url}/VW0/"
    download_raw_data(
        vw0_path, channel_of_interest, raw_data_path, log_s3_path=log_s3_path
    )

    # compute stitching alignments first if you need to
    # download stitching files if they exist at log path
    if (
        not download_terastitcher_files(log_s3_path, raw_data_path)
        and channel_of_interest == 0
    ):
        metadata = run_terastitcher(
            raw_data_path,
            stitched_data_path,
            input_s3_path,
            log_s3_path=log_s3_path,
            compute_only=True,
        )

    # bias correct all tiles
    # save bias correction tile to log_s3_path
    correct_raw_data(raw_data_path, channel_of_interest, log_s3_path=log_s3_path)

    # now stitch the data with alignments we computed
    metadata = run_terastitcher(
        raw_data_path,
        stitched_data_path,
        input_s3_path,
        log_s3_path=log_s3_path,
        stitch_only=True,
    )

    # downsample and upload stitched data to S3
    stitched_path = glob(f"{stitched_data_path}/RES*")[0]
    create_precomputed_volume(
        stitched_path, np.array(metadata["voxel_size"]), output_s3_path
    )

    # correct whole brain bias
    # in order to not replicate data (higher S3 cost)
    # overwrite original precomputed volume with corrected data
    correct_stitched_data(output_s3_path, output_s3_path)

    # print viz link to console
    # visualize data at 5 microns
    viz_link = create_viz_link(
        [output_s3_path], output_resolution=np.array([5] * 3) / 1e6
    )
    print("###################")
    print(f"VIZ LINK: {viz_link}")
    print("###################")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Run COLM pipeline including bias correction, stitching, upoad to S3"
    )
    parser.add_argument(
        "input_s3_path",
        help="S3 path to input colm data. Should be of the form s3://<bucket>/<experiment>",
        type=str,
    )
    parser.add_argument(
        "output_s3_path",
        help="S3 path to store precomputed volume. Precomputed volumes for each channel will be stored under this path. Should be of the form s3://<bucket>/<path_to_precomputed>. The data will be saved at s3://<bucket>/<path_to_precomputed>/CHN0<channel>",
        type=str,
    )
    # parser.add_argument('channel_of_interest', help='Channel of interest in experiment',  type=int)
    parser.add_argument(
        "num_channels", help="Number of channels in experiment", type=int
    )
    parser.add_argument(
        "autofluorescence_channel", help="Autofluorescence channel number.", type=int
    )
    parser.add_argument(
        "--raw_data_path",
        help="Local path where corrected raw data will be stored.",
        type=str,
        default=os.path.expanduser("~/ssd1"),
    )
    parser.add_argument(
        "--stitched_data_path",
        help="Local path where stitched slices will be stored.",
        type=str,
        default=os.path.expanduser("~/ssd2"),
    )
    parser.add_argument(
        "--log_s3_path",
        help="S3 path at which pipeline intermediates can be stored including bias correctin tile.",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    # for all channels in experiment
    for i in range(args.num_channels):
        output_s3_path = args.output_s3_path.strip("/")
        colm_pipeline(
            args.input_s3_path,
            f"{output_s3_path}/CHN0{i}",
            i,
            args.autofluorescence_channel,
            args.raw_data_path,
            args.stitched_data_path,
            args.log_s3_path,
        )
        if i < args.num_channels - 1:
            # delete all tiff files in raw_data_path
            directories_to_remove = glob(f"{args.raw_data_path}/LOC*")
            directories_to_remove.extend(glob(f"{args.stitched_data_path}/RES*"))
            with tqdm_joblib(
                tqdm(
                    desc=f"Delete files from CHN0{i}", total=len(directories_to_remove)
                )
            ) as progress_bar:
                Parallel(-1)(delayed(shutil.rmtree)(f) for f in directories_to_remove)
            # make sure to delete mdata.bin from terastitcher
            os.remove(f"{args.raw_data_path}/mdata.bin")
