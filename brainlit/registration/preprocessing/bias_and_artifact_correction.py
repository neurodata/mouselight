import numpy as np
from numpy import ma
import SimpleITK as sitk
from scipy.ndimage.filters import gaussian_filter
from skimage import morphology
from skimage.filters import threshold_otsu
from skimage.transform import resize, rescale

from ..lddmm._lddmm_utilities import _validate_ndarray


def correct_bias_field(image, correct_at_scale=1, as_float32=True, **kwargs):
    """
    Shifts image such that its minimum value is 1, computes the bias field after downsampling by correct_at_scale,
    upsamples this bias field and applies it to the shifted image, then undoes the shift and returns the result.
    Computes bias field using sitk.N4BiasFieldCorrection (http://bit.ly/2oFwAun).

    Args:
        image (np.ndarray): The image to be bias corrected.
        correct_at_scale (float, optional): The scale by which the shape of image is reduced before computing the bias. Defaults to 1.
        as_float32 (bool, optional): If True, image is internally cast as a sitk.Image of type sitkFloat32. If False, it is of type sitkFloat64. Defaults to True.

    Kwargs:
        Any additional keyword arguments overwrite the default values passed to sitk.N4BiasFieldCorrection.

    Returns:
        np.ndarray: A copy of image after bias correction.
    """

    # Validate inputs.

    # Validate image.
    image = _validate_ndarray(image, dtype=float)

    # Verify correct_at_scale.
    correct_at_scale = float(correct_at_scale)
    if correct_at_scale < 1:
        raise ValueError(
            f"correct_at_scale must be equal to or greater than 1.\n"
            f"correct_at_scale: {correct_at_scale}."
        )

    # Shift image such that its minimum value lies at 1.
    image_min = image.min()
    image = image - image_min + 1

    # Downsample image according to scale.
    downsampled_image = rescale(image, correct_at_scale)

    # Bias correct downsampled_image.
    N4BiasFieldCorrection_kwargs = dict(
        image=downsampled_image,
        maskImage=np.ones_like(downsampled_image),
        convergenceThreshold=0.001,
        maximumNumberOfIterations=[50, 50, 50, 50],
        biasFieldFullWidthAtHalfMaximum=0.15,
        wienerFilterNoise=0.01,
        numberOfHistogramBins=200,
        numberOfControlPoints=[4, 4, 4],
        splineOrder=3,
        useMaskLabel=True,
        maskLabel=1,
    )
    # Overwrite default arguments with user-supplied kwargs.
    N4BiasFieldCorrection_kwargs.update(kwargs)
    # Convert image and maskImage N4BiasFieldCorrection_kwargs from type np.ndarray to type sitk.Image.
    sitk_image = sitk.GetImageFromArray(N4BiasFieldCorrection_kwargs["image"])
    sitk_image = sitk.Cast(
        sitk_image, sitk.sitkFloat32 if as_float32 else sitk.sitkFloat64
    )
    sitk_maskImage = N4BiasFieldCorrection_kwargs["maskImage"].astype(np.uint8)
    sitk_maskImage = sitk.GetImageFromArray(sitk_maskImage)
    N4BiasFieldCorrection_kwargs.update(
        image=sitk_image,
        maskImage=sitk_maskImage,
    )
    bias_corrected_downsampled_image = sitk.N4BiasFieldCorrection(
        *N4BiasFieldCorrection_kwargs.values()
    )
    bias_corrected_downsampled_image = sitk.GetArrayFromImage(
        bias_corrected_downsampled_image
    )

    # Compute bias from bias_corrected_downsampled_image.
    downsample_computed_bias = bias_corrected_downsampled_image / downsampled_image

    # Upsample bias.
    upsampled_bias = resize(downsample_computed_bias, image.shape)

    # Apply upsampled bias to original resolution shifted image.
    bias_corrected_image = image * upsampled_bias

    # Reverse the initial shift.
    bias_corrected_image += image_min - 1

    return bias_corrected_image


def remove_grid_artifact(
    image,
    z_axis=0,
    sigma_blur=None,
    mask="Otsu",
    otsu_nbins=256,
    otsu_binary_closing_radius=None,
    otsu_background_is_dim=True,
):
    """
    Remove the grid artifact from tiled data.

    Args:
        image (np.ndarray): The image with a grid artifact.
        z_axis (int, optional): The axis along which the tiles are stacked. Defaults to 0.
        sigma_blur (float, optional): The size of the blur used to compute the bias for grid edges in units of voxels.
            Should be approximately the size of the tiles. Defaults to np.ceil(np.sqrt(np.min(image.shape))).
        mask (np.ndarray, str, NoneType, optional): A mask of the valued voxels in the image.
            Supported values are:
                a np.ndarray with a shape corresponding to image.shape,
                None, indicating no mask (i.e. all voxels are considered in the artifact correction),
                'Otsu', indicating that the Otsu threshold will be used to identify foreground and background voxels.
            Defaults to 'Otsu'.
        otsu_nbins (int, optional): The number of bins used to calculate the histogram in skimage.filters.threshold_otsu if mask == 'Otsu'. Defaults to 256.
        otsu_binary_closing_radius (int, optional): The radius of the structuring element given to binary_close if mask == 'Otsu'. Defaults to np.ceil(np.sqrt(np.min(image.shape)) / 3).
        otsu_background_is_dim (bool, optional): If True and mask == 'Otsu', when computing the mask it is assumed that the background will have a lower value than the foreground. Defaults to True.

    Returns:
        np.ndarray: A copy of image with its grid artifact removed.
    """

    # Validate inputs.

    # Validate image.
    image = _validate_ndarray(image, dtype=float)

    # Validate sigma_blur.
    sigma_blur = (
        float(sigma_blur)
        if sigma_blur is not None
        else np.ceil(np.sqrt(np.min(image.shape)))
    )

    # Validate otsu_binary_closing_radius.
    otsu_binary_closing_radius = (
        int(otsu_binary_closing_radius)
        if otsu_binary_closing_radius is not None
        else int(np.ceil(np.sqrt(np.min(image.shape)) / 3))
    )

    # Validate otsu_background_is_dim.
    otsu_background_is_dim = bool(otsu_background_is_dim)

    # Construct masked_image as a ma.MaskedArray.

    # Interpret input mask.
    if mask is None:
        mask = np.ones_like(image, bool)
    elif isinstance(mask, str) and mask == "Otsu":
        # Finds the optimal split threshold between the foreground anad background,
        # by maximizing the interclass variance and minimizing the intraclass variance between voxel intensities,
        # with he higher-intensity class labeled as 1.
        otsu_threshold = threshold_otsu(image, nbins=otsu_nbins)
        mask = np.ones_like(image, int)
        # Segment image by otsu_threshold.
        mask[
            image <= otsu_threshold
            if otsu_background_is_dim
            else image >= otsu_threshold
        ] = 0
        # Perform binary closing and then binary fill hole on image to remove mislabed background voxels inside the foreground regions.
        mask = sitk.GetArrayFromImage(
            sitk.BinaryFillhole(
                sitk.BinaryMorphologicalClosing(
                    sitk.GetImageFromArray(mask),
                    otsu_binary_closing_radius,
                    sitk.sitkBall,
                )
            )
        ).astype(bool)
    else:
        mask = _validate_ndarray(mask, reshape_to_shape=image.shape, dtype=bool)

    # Use the inverse of mask to create masked_image.
    masked_image = ma.masked_array(image, mask=~mask)

    # Shift masked_image so that its minimum lies at 1.
    masked_image_min = np.min(masked_image)
    masked_image -= masked_image_min - 1

    # Correct grid artifacts.

    # Take the average across z
    mean_across_z = np.mean(masked_image, axis=z_axis, keepdims=True)

    # Blur the mean with a gaussian.
    z_projection_bias = gaussian_filter(mean_across_z, sigma_blur) / mean_across_z

    # Apply the z_projection_bias to correct the masked_image.
    corrected_masked_image = masked_image * z_projection_bias

    # Reverse the shift to restore the original data window.
    corrected_masked_image += masked_image_min - 1

    # Return the unmasked array.
    return corrected_masked_image.data
