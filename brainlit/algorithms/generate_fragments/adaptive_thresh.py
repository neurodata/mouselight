# Reference: http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/300_Segmentation_Overview.html

import SimpleITK as sitk
from sklearn.mixture import GaussianMixture
import numpy as np


def get_seed(voxel):
    """
    Get a seed point for the center of a brain volume.

    Parameters
    ----------
    voxel : tuple:
        The seed coordinates in x y z.

    Returns
    -------
    tuple
        A tuple containing the (x, y, z)-coordinates of the seed.

    """
    numpy_seed = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
    sitk_seed = (int(voxel[2]), int(voxel[1]), int(voxel[0]))
    return numpy_seed, sitk_seed


def get_img_T1(img):
    """
    Converts a volume cutout to a SimpleITK image, as wel
    as a SimpleITK image with scaled intensity values to 0-255.

    Parameters
    ----------
    img : cloudvolume.volumecutout.VolumeCutout
        The volume to convert to a SimpleITK image.

    Returns
    -------
    img_T1 : SimpleITK.SimpleITK.Image
        A SimpleITK image.

    img_T1_255 : SimpleITK.SimpleITK.Image
        A SimpleITK image with
        intensity values between 0 and 255 inclusive.

    """

    img_T1 = sitk.GetImageFromArray(np.squeeze(img), isVector=False)
    img_T1_255 = sitk.Cast(sitk.RescaleIntensity(img_T1), sitk.sitkUInt8)
    return img_T1, img_T1_255


def thres_from_gmm(img, random_seed=2):
    """
    Computes a numerical threshold for segmentation based
    on a 2-Component Gaussian mixture model.

    The threshold is the minimum value included in the Gaussian
    mixture model-component containning the highest intensity value.

    Parameters
    ----------
    img : cloudvolume.volumecutout.VolumeCutout
        The image or volume to threshold.

    random_seed : int
        The random seed for the Gaussian mixture model.

    Returns
    -------
    int
        The threshold value.

    """

    _, img_T1_255 = get_img_T1(img)
    img_array = sitk.GetArrayFromImage(img_T1_255)
    flat_array = img_array.flatten().reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=random_seed)
    y = gmm.fit_predict(flat_array)
    mask = y.astype(bool)
    a = flat_array[mask]
    b = flat_array[~mask]
    if a.max() > b.max():
        thres = a.min()
    else:
        thres = b.min()

    return int(thres)


def fast_marching_seg(img, seed, stopping_value=150, sigma=0.5):
    """
    Computes a fast-marching segmentation.

    Parameters
    ----------
    img : cloudvolume.volumecutout.VolumeCutout
        The volume to segment.

    seed : tuple
        The seed containing a coordinate within a known segment.

    stopping_value : float
        The algorithm stops when the value of the smallest trial
        point is greater than this stopping value.

    sigma : float
        Sigma used in computing the feature image.

    Returns
    -------
    labels : numpy.ndarray
        An array consisting of the pixelwise segmentation.

    """

    img_T1, img_T1_255 = get_img_T1(img)
    feature_img = sitk.GradientMagnitudeRecursiveGaussian(img_T1_255, sigma=sigma)
    speed_img = sitk.BoundedReciprocal(feature_img)
    fm_filter = sitk.FastMarchingBaseImageFilter()
    fm_filter.SetTrialPoints([seed])
    fm_filter.SetStoppingValue(stopping_value)
    fm_img = fm_filter.Execute(speed_img)
    fm_img = sitk.Cast(sitk.RescaleIntensity(fm_img), sitk.sitkUInt8)
    labels = sitk.GetArrayFromImage(fm_img)
    labels = (~labels.astype(bool)).astype(int)
    return labels


def level_set_seg(
    img,
    seed,
    lower_threshold=None,
    upper_threshold=None,
    factor=2,
    max_rms_error=0.02,
    num_iter=1000,
    curvature_scaling=0.5,
    propagation_scaling=1,
):
    """
    Computes a level-set segmentation.

    When root mean squared change in the level set function for an iteration is below
    the threshold, or the maximum number of iteration have elapsed,
    the algorithm is said to have converged.

    Parameters
    ----------
    img : cloudvolume.volumecutout.VolumeCutout
        The volume to segment.

    seed : tuple
        The seed containing a coordinate within a known segment.

    lower_threshold : float
        The lower threshold for segmentation. Set based on image statistics if None.

    upper_threshold : float
        The upper threshold for segmentation. Set based on image statistics if None.

    factor : float
        The scaling factor on the standard deviation used in computing thresholds
        from image statistics.

    max_rms_error : float
        Root mean squared convergence criterion threshold.

    num_iter : int
        Maximum number of iterations.

    curvature_scaling : float
        Curvature scaling for the segmentation.

    propagation_scaling : float
        Propagation scaling for the segmentation.

    Returns
    -------
    labels : numpy.ndarray
        An array consisting of the pixelwise segmentation.

    """

    img_T1, img_T1_255 = get_img_T1(img)

    seg = sitk.Image(img_T1_255.GetSize(), sitk.sitkUInt8)
    seg.CopyInformation(img_T1_255)
    seg[seed] = 1
    seg = sitk.BinaryDilate(seg, 1)

    stats = sitk.LabelStatisticsImageFilter()
    stats.Execute(img_T1_255, seg)

    if lower_threshold == None:
        lower_threshold = stats.GetMean(1) - factor * stats.GetSigma(1)
    if upper_threshold == None:
        upper_threshold = stats.GetMean(1) + factor * stats.GetSigma(1)

    init_ls = sitk.SignedMaurerDistanceMap(
        seg, insideIsPositive=True, useImageSpacing=True
    )

    lsFilter = sitk.ThresholdSegmentationLevelSetImageFilter()
    lsFilter.SetLowerThreshold(lower_threshold)
    lsFilter.SetUpperThreshold(upper_threshold)
    lsFilter.SetMaximumRMSError(max_rms_error)
    lsFilter.SetNumberOfIterations(num_iter)
    lsFilter.SetCurvatureScaling(curvature_scaling)
    lsFilter.SetPropagationScaling(propagation_scaling)
    lsFilter.ReverseExpansionDirectionOn()
    ls = lsFilter.Execute(init_ls, sitk.Cast(img_T1_255, sitk.sitkFloat32))

    labels = sitk.GetArrayFromImage(ls > 0)
    return labels


def connected_threshold(img, seed, lower_threshold=None, upper_threshold=255):
    """
    Compute a threshold-based segmentation via connected region growing.

    Labelled pixels are connected to a seed and lie within a range of values.

    Parameters
    ----------
    img : cloudvolume.volumecutout.VolumeCutout
        The volume to segment.

    seed : tuple
        The seed containing a coordinate within a known segment.

    lower_threshold : float
        The lower threshold for the region growth.
        Set by a 2-component Gaussian mixture model if None.

    upper_threshold : float
        The upper threshold for the region growth.

    Returns
    -------
    labels : numpy.ndarray
        An array consisting of the pixelwise segmentation.

    """

    img_T1, img_T1_255 = get_img_T1(img)
    seg = sitk.Image(img_T1.GetSize(), sitk.sitkUInt8)
    seg.CopyInformation(img_T1)
    # seg[seed] = 1
    for s in seed:
        seg[s] = 1
    seg = sitk.BinaryDilate(seg, 1)

    if lower_threshold == None:
        lower_threshold = thres_from_gmm(img)

    seg_con = sitk.ConnectedThreshold(
        img_T1_255, seedList=seed, lower=lower_threshold, upper=upper_threshold
    )

    vectorRadius = (1, 1, 1)
    kernel = sitk.sitkBall
    seg_clean = sitk.BinaryMorphologicalClosing(seg_con, vectorRadius, kernel)

    labels = sitk.GetArrayFromImage(seg_clean)
    return labels


def confidence_connected_threshold(
    img, seed, num_iter=1, multiplier=1, initial_neighborhood_radius=1, replace_value=1
):
    """
    Compute a threshold-based segmentation via confidence-connected region growing.

    The segmentation is based on pixels with intensities that are consistent
    with pixel statistics of a seed point.
    Pixels connected to the seed point with values within a confidence interval
    are grouped.
    The confidence interval is the mean plus of minus the "multiplier" times
    the standard deviation.
    After an initial segmentation is completed, the mean and standard deviation
    are calculated again at each iteration using pixels in the previous segmentation.

    Parameters
    ----------
    img : cloudvolume.volumecutout.VolumeCutout
        The volume to segment.

    seed : tuple
        The seed containing a coordinate within a known segment.

    num_iter : int
        The number of iterations to run the algorithm.

    multiplier : float
        Multiplier for the confidence interval.

    initial_neighborhood_radius : int
        The initial neighborhood radius for computing statistics on the seed pixel.

    replace_value : int
        The value to replace thresholded pixels.

    Returns
    -------
    labels : numpy.ndarray
        An array consisting of the pixelwise segmentation.

    """

    img_T1, img_T1_255 = get_img_T1(img)
    seg = sitk.Image(img_T1.GetSize(), sitk.sitkUInt8)
    seg.CopyInformation(img_T1)
    # seg[seed] = 1
    for s in seed:
        seg[s] = 1
    seg = sitk.BinaryDilate(seg, 1)

    seg_con = sitk.ConfidenceConnected(
        img_T1_255,
        seedList=seed,
        numberOfIterations=num_iter,
        multiplier=multiplier,
        initialNeighborhoodRadius=initial_neighborhood_radius,
        replaceValue=replace_value,
    )

    vectorRadius = (1, 1, 1)
    kernel = sitk.sitkBall
    seg_clean = sitk.BinaryMorphologicalClosing(seg_con, vectorRadius, kernel)

    labels = sitk.GetArrayFromImage(seg_clean)
    return labels


def neighborhood_connected_threshold(
    img, seed, lower_threshold=None, upper_threshold=255
):
    """
    Compute a threshold-based segmentation via neighborhood-connected region growing.

    Labelled pixels are connected to a seed and lie within a neighborhood.

    Parameters
    ----------
    img : cloudvolume.volumecutout.VolumeCutout
        The volume to segment.

    seed : tuple
        The seed containing a coordinate within a known segment.

    lower_threshold : float
        The lower threshold for the region growth.
        Set by a 2-component Gaussian mixture model if None.

    upper_threshold : float
        The upper threshold for the region growth.

    Returns
    -------
    labels : numpy.ndarray
        An array consisting of the pixelwise segmentation.

    """

    img_T1, img_T1_255 = get_img_T1(img)
    seg = sitk.Image(img_T1.GetSize(), sitk.sitkUInt8)
    seg.CopyInformation(img_T1)
    for s in seed:
        seg[s] = 1
    seg = sitk.BinaryDilate(seg, 1)

    if lower_threshold == None:
        lower_threshold = thres_from_gmm(img)

    seg_con = sitk.NeighborhoodConnected(
        img_T1_255, seedList=seed, lower=lower_threshold, upper=upper_threshold
    )

    vectorRadius = (1, 1, 1)
    kernel = sitk.sitkBall
    seg_clean = sitk.BinaryMorphologicalClosing(seg_con, vectorRadius, kernel)

    labels = sitk.GetArrayFromImage(seg_clean)
    return labels


def otsu(img, seed):
    """
    Compute a threshold-based segmentation via Otsu's method.

    Parameters
    ----------
    img : cloudvolume.volumecutout.VolumeCutout
        The volume to segment.

    Returns
    -------
    labels : numpy.ndarray
        An array consisting of the pixelwise segmentation.

    """

    img_T1, img_T1_255 = get_img_T1(img)

    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    seg = otsu_filter.Execute(img_T1_255)
    labels = sitk.GetArrayFromImage(seg)
    if labels[seed] != 1:
        labels = abs(labels - 1)
    return labels


def gmm_seg(img, seed, random_seed=3):
    """
    Compute a threshold-based segmentation via a 2-component Gaussian mixture model.

    Parameters
    ----------
    img : cloudvolume.volumecutout.VolumeCutout
        The volume to segment.

    random_seed : int
        The random seed for the Gaussian mixture model.

    Returns
    -------
    labels : numpy.ndarray
        An array consisting of the pixelwise segmentation.

    """

    img_T1, img_T1_255 = get_img_T1(img)
    img_array = sitk.GetArrayFromImage(img_T1_255)
    flat_array = img_array.flatten().reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=random_seed)
    y = gmm.fit_predict(flat_array)
    labels = y.reshape(img.shape).squeeze()
    if labels[seed] != 1:
        labels = abs(labels - 1)
    return labels
