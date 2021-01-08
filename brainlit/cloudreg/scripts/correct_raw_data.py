import glob
import time
import os
from io import BytesIO
import argparse
import boto3
from botocore.client import Config
import numpy as np
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count
import math
import tifffile as tf
import SimpleITK as sitk
from util import (
    tqdm_joblib,
    chunks,
    imgResample,
    upload_file_to_s3,
    S3Url,
    s3_object_exists,
    get_bias_field,
)


config = Config(connect_timeout=5, retries={"max_attempts": 5})


def sum_tiles(files):
    raw_tile = np.squeeze(tf.imread(files[0]))
    running_sum = np.zeros(raw_tile.shape, dtype="float")

    for f in files:
        running_sum += np.squeeze(tf.imread(f))

    return running_sum


def correct_tile(raw_tile_path, outdir, bias, background_value=None):
    # overwrite existing tile
    out_path = raw_tile_path
    raw_tile = np.squeeze(tf.imread(raw_tile_path)).astype("float")

    if bias is None:
        tf.imwrite(out_path, data=raw_tile.astype("uint16"), compress=3, append=False)

    else:
        # rescale corrected tile to be uint16
        # for Terastitcher
        tile_bc = np.clip(raw_tile - background_value, 0, None)
        corrected_tile = np.around(tile_bc * bias)
        # clip values above uint16.max and below 0
        # corrected_tile = np.clip(corrected_tile, 0, np.iinfo(np.uint16).max)
        # corrected_tile = (corrected_tile/(2**12 - 1)) * np.iinfo('uint16').max
        tf.imwrite(
            out_path, data=corrected_tile.astype("uint16"), compress=3, append=False
        )


def correct_tiles(tiles, outdir, bias, background_value=None):
    for tile in tiles:
        correct_tile(tile, outdir, bias, background_value)


def get_background_value(raw_data_path):
    first_plane = np.sort(glob.glob(f"{raw_data_path}/LOC*/*PLN0000*.tiff"))

    def _mean(image_path):
        return np.mean(tf.imread(image_path))

    means = Parallel(-1)(
        delayed(_mean)(tile)
        for tile in tqdm(first_plane, desc="Finding background tile...")
    )
    # pat to tile with lowest mean
    # background_tile_path = first_plane[np.argmin(means)]
    # return mean of that tile
    return np.min(means)


def correct_raw_data(
    raw_data_path,
    channel,
    subsample_factor=2,
    log_s3_path=None,
    background_correction=True,
):

    total_n_jobs = cpu_count()
    # overwrite existing raw data with corrected data
    outdir = raw_data_path

    # get list of all tiles to correct for  given channel
    all_files = np.sort(glob.glob(f"{raw_data_path}/*/*.tiff"))
    if background_correction:
        background_val = get_background_value(raw_data_path)
    total_files = len(all_files)

    bias_path = f"{outdir}/CHN0{channel}_bias.tiff"
    if os.path.exists(bias_path):
        bias = tf.imread(bias_path)

    else:
        # subsample tiles
        files_cb = all_files[::subsample_factor]
        num_files = len(files_cb)

        # compute running sums in parallel
        sums = Parallel(total_n_jobs, verbose=10)(
            delayed(sum_tiles)(f)
            for f in chunks(files_cb, math.ceil(num_files // (total_n_jobs)) + 1)
        )
        sums = [i[:, :, None] for i in sums]
        mean_tile = np.squeeze(np.sum(np.concatenate(sums, axis=2), axis=2)) / num_files
        if background_correction:
            # subtract background out from bias correction
            mean_tile -= background_val
        mean_tile = sitk.GetImageFromArray(mean_tile)

        # get the bias correction tile using N4ITK
        bias = sitk.GetArrayFromImage(get_bias_field(mean_tile, scale=1.0))

        # save bias tile to local directory
        tf.imsave(bias_path, bias.astype("float32"))

    # save bias tile to S3
    if log_s3_path:
        s3 = boto3.resource("s3")
        img = Image.fromarray(bias)
        fp = BytesIO()
        img.save(fp, format="TIFF")
        # reset pointer to beginning  of file
        fp.seek(0)
        log_s3_url = S3Url(log_s3_path.strip("/"))
        bias_path = f"{log_s3_url.key}/CHN0{channel}_bias.tiff"
        s3.Object(log_s3_url.bucket, bias_path).upload_fileobj(fp)

    # correct all the files and save them
    files_per_proc = math.ceil(total_files / total_n_jobs) + 1
    work = chunks(all_files, files_per_proc)
    with tqdm_joblib(tqdm(desc="Correcting tiles", total=total_n_jobs)) as progress_bar:
        Parallel(n_jobs=total_n_jobs, verbose=10)(
            delayed(correct_tiles)(files, outdir, bias, background_val)
            for files in work
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-raw_data_path", help="Path to raw data in COLM format.", type=str
    )
    parser.add_argument(
        "--log_s3_path", help="S3 path where bias correction will live.", type=str
    )
    parser.add_argument(
        "-channel",
        help="Channel number to process. accepted values are 0, 1, or 2",
        type=str,
    )
    # parser.add_argument('--outdir', help='Path to output directory to store corrected tiles. VW0 directory will  be saved here. Default: ~/', default='/home/ubuntu/' ,type=str)
    parser.add_argument(
        "--subsample_factor",
        help="Factor to subsample the tiles by to compute the bias. Default is subsample by 5 which means every 5th tile  will be used.",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--background_tile",
        help="Local path to tile that will be used for background subtraction.",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    correct_raw_data(
        args.raw_data_path, args.channel, args.subsample_factor, args.log_s3_path
    )
