import numpy as np
from skimage.transform import resize
from scipy.interpolate import interpn

from ..lddmm._lddmm_utilities import _validate_scalar_to_multi
from ..lddmm._lddmm_utilities import _validate_ndarray
from ..lddmm._lddmm_utilities import _validate_resolution
from ..lddmm._lddmm_utilities import _compute_axes
from ..lddmm._lddmm_utilities import _compute_coords


def _resample(image, resolution, final_resolution, **interpn_kwargs):
    # interpn wrapper, not to be used without achievable final_resolution.

    if not np.all(image.shape * resolution % final_resolution):
        raise ValueError(
            f"final_resolution must evenly fit into the real shape of image."
        )

    final_shape = np.array(image.shape * resolution / final_resolution, int)

    # Validate inputs.
    image = _validate_ndarray(image)
    resolution = _validate_resolution(resolution, image.ndim)
    desired_resolution = _validate_resolution(desired_resolution, image.ndim)

    # Construct arguments for scipy.interpolate.interpn.
    real_axes = _compute_axes(shape=image.shape, resolution=resolution)
    new_real_coords = _compute_coords(shape=final_shape, resolution=final_resolution)

    return interpn(points=real_axes, values=image, xi=new_real_coords, **interpn_kwargs)


def change_resolution_to(
    image,
    resolution,
    desired_resolution,
    pad_to_match_res=True,
    err_to_higher_res=True,
    return_final_resolution=False,
    **resize_kwargs,
):

    # Validate inputs.
    image = _validate_ndarray(image)
    resolution = _validate_resolution(resolution, image.ndim)
    desired_resolution = _validate_resolution(desired_resolution, image.ndim)

    # Compute final_shape and final_resolution.

    # Take the exact floating point final_shape that would be necessary to achieve desired_resolution.
    final_shape = image.shape * resolution / desired_resolution

    # Adjust final_shape and compute final_resolution according to specification.
    if pad_to_match_res:
        # Guarantee realization of desired_resolution at the possible expense of maintaining the true shape (shape * resolution).
        # Note: "true shape" implies real-valued shape: the product of image shape and corresponding resolution.
        final_shape = np.ceil(final_shape)
        # Pad image evenly until image.shape * resolution >= final_shape * desired_resolution.
        minimum_image_padding = np.ceil(
            (final_shape * desired_resolution - image.shape * resolution) / resolution
        )
        pad_width = np.array(
            list(
                zip(
                    np.ceil(minimum_image_padding / 2),
                    np.ceil(minimum_image_padding / 2),
                )
            ),
            int,
        )
        old_true_shape = resolution * image.shape
        new_true_shape = desired_resolution * final_shape
        stat_length = np.maximum(
            1,
            np.ceil(
                (desired_resolution - ((new_true_shape - old_true_shape) / 2))
                / resolution
            ),
        ).astype(int)
        stat_length = np.broadcast_to(stat_length, pad_width.T.shape).T
        image = np.pad(
            image, pad_width=pad_width, mode="mean", stat_length=stat_length
        )  # Side effect: breaks alias.
        # final_resolution has been guaranteed to equal desired_resolution.
        final_resolution = desired_resolution
    else:
        # Guarantee the true shape (image.shape * resolution) is maintained at the possible expense of achieving desired_resolution.
        if err_to_higher_res:
            # Round resolution up.
            final_shape = np.ceil(final_shape)
        else:
            # Round resolution down.
            final_shape = np.floor(final_shape)
        # Compute the achieved resultant resolution, or final_resolution.
        final_resolution = image.shape * resolution / final_shape
        # Warn the user if desired_resolution cannot be produced from image and resolution.
        if not np.array_equal(
            final_shape, image.shape * resolution / desired_resolution
        ):  # If desired_resolution != final_resolution.
            warnings.warn(
                message=f"Could not exactly produce the desired_resolution.\n"
                f"xyz_resolution {xyz_resolution}.\n"
                f"desired_resolution: {desired_resolution}.\n"
                f"final_resolution: {final_resolution}.",
                category=RuntimeWarning,
            )

    # Note: if the function used for resampling below does not include anti-aliasing in the dase of downsampling,
    # that ought to be performed here. skimage.transform.resize does perform anti-aliasing by default.

    # Perform resampling.

    resampled_image = resize(image, final_shape, **resize_kwargs)
    return resampled_image


def change_resolution_by(
    image,
    scales,
    resolution=1,
    pad_to_match_res=True,
    err_to_higher_res=True,
    return_final_resolution=False,
    **resize_kwargs,
):

    image = _validate_ndarray(image)
    scales = _validate_scalar_to_multi(scales, size=image.ndim)
    resolution = _validate_resolution(resolution, image.ndim)
    desired_resolution = resolution / scales

    change_resolution_to_kwargs = dict(
        image=image,
        resolution=resolution,
        desired_resolution=desired_resolution,
        pad_to_match_res=pad_to_match_res,
        err_to_higher_res=err_to_higher_res,
        return_final_resolution=return_final_resolution,
        **resize_kwargs,
    )

    return change_resolution_to(**change_resolution_to_kwargs)
