import argparse
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
from cloudvolume import CloudVolume
import tinybrain
from joblib import Parallel, delayed

from util import imgResample, tqdm_joblib, get_bias_field


def process_slice(bias_slice, z, data_orig_path, data_bc_path):
    data_vol = CloudVolume(
        data_orig_path, parallel=False, progress=False, fill_missing=True
    )
    data_vol_bc = CloudVolume(
        data_bc_path, parallel=False, progress=False, fill_missing=True
    )
    data_vols_bc = [
        CloudVolume(data_bc_path, mip=i, parallel=False)
        for i in range(len(data_vol_bc.scales))
    ]
    # convert spcing rom nm to um
    new_spacing = np.array(data_vol.scales[0]["resolution"][:2]) / 1000
    bias_upsampled_sitk = imgResample(
        bias_slice, new_spacing, size=data_vol.scales[0]["size"][:2]
    )
    bias_upsampled = sitk.GetArrayFromImage(bias_upsampled_sitk)
    data_native = np.squeeze(data_vol[:, :, z]).T
    data_corrected = data_native * bias_upsampled
    img_pyramid = tinybrain.downsample_with_averaging(
        data_corrected.T[:, :, None],
        factor=(2, 2, 1),
        num_mips=len(data_vol_bc.scales) - 1,
    )
    data_vol_bc[:, :, z] = data_corrected.T.astype("uint16")[:, :, None]
    for i in range(len(data_vols_bc) - 1):
        data_vols_bc[i + 1][:, :, z] = img_pyramid[i].astype("uint16")


def correct_stitched_data(data_s3_path, out_s3_path, resolution=15, num_procs=12):
    # create vol
    vol = CloudVolume(data_s3_path)
    mip = 0
    for i in range(len(vol.scales)):
        # get low res image smaller than 15 um
        if vol.scales[i]["resolution"][0] <= resolution * 1000:
            mip = i
    vol_ds = CloudVolume(data_s3_path, mip, parallel=True, fill_missing=True)

    # create new vol if it doesnt exist
    vol_bc = CloudVolume(out_s3_path, info=vol.info.copy())
    vol_bc.commit_info()

    # download image at low res
    data = sitk.GetImageFromArray(np.squeeze(vol_ds[:, :, :]).T)
    data.SetSpacing(np.array(vol_ds.scales[mip]["resolution"]) / 1000)

    bias = get_bias_field(data, scale=0.125)
    bias_slices = [bias[:, :, i] for i in range(bias.GetSize()[-1])]
    try:
        with tqdm_joblib(
            tqdm(desc=f"Uploading bias corrected data...", total=len(bias_slices))
        ) as progress_bar:
            Parallel(num_procs, timeout=3600, verbose=10)(
                delayed(process_slice)(bias_slice, z, data_s3_path, out_s3_path)
                for z, bias_slice in enumerate(bias_slices)
            )
    except:
        print("timed out on bias correcting slice. moving to next step.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Correct whole brain bias field in image at native resolution."
    )
    parser.add_argument(
        "data_s3_path",
        help="full s3 path to data of interest as precomputed volume. must be of the form `s3://bucket-name/path/to/channel`",
    )
    parser.add_argument("out_s3_path", help="S3 path to save output results")
    parser.add_argument(
        "--num_procs", help="number of processes to use", default=15, type=int
    )
    parser.add_argument(
        "--resolution",
        help="max resolution for computing bias correction in microns",
        default=15,
        type=float,
    )
    args = parser.parse_args()

    correct_stitched_data(
        args.data_s3_path, args.out_s3_path, args.resolution, args.num_procs
    )
