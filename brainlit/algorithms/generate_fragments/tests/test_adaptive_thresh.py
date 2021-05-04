import pytest
from brainlit.algorithms.generate_fragments.adaptive_thresh import (
    get_seed,
    get_img_T1,
    thres_from_gmm,
    fast_marching_seg,
    level_set_seg,
    connected_threshold,
    confidence_connected_threshold,
    neighborhood_connected_threshold,
    otsu,
    gmm_seg,
)
import SimpleITK as sitk
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

##################
### validation ###
##################


def test_get_seed():

    # define voxel
    voxel = (10.131, 30.6001, 100)
    numpy_seed, sitk_seed = get_seed(voxel)
    assert numpy_seed == (10, 30, 100)
    assert sitk_seed == (100, 30, 10)


def test_get_img_T1():
    img = np.array([[[100, 250], [800, 300]], [[1200, 2000], [3000, 2500]]])
    img_T1, img_T1_255 = get_img_T1(img)
    assert type(img_T1) == sitk.Image
    assert type(img_T1_255) == sitk.Image
    assert img_T1_255.GetPixelIDTypeAsString() == "8-bit unsigned integer"


def test_thres_from_gmm():
    # define two groups of Gaussian distribution points with distinct mean values
    Good1 = False
    Good2 = False
    # to ensure the random number does not fall below 0 (the minimum value of an 8-bit image)
    while Good1 == False:
        G1 = np.round(np.random.normal(loc=40, scale=10, size=(499, 1)))
        if min(G1) > 0:
            Good1 = True

    # to ensure the random number does not exceed 255 (the maximum value of an 8-bit image)
    while Good2 == False:
        G2 = np.round(np.random.normal(loc=220, scale=10, size=(499, 1)))
        if max(G2) < 255:
            Good2 = True

    # the minimum value of the high-mean Gaussian distribution determines the threshold
    thre_predicted = np.nanmin(G2)
    # construct a 3D image with the two groups of points
    img = np.append(np.concatenate((G1, G2)), np.array([[0.0], [255.0]])).reshape(
        (10, 10, 10)
    )
    # calculate the threshold with `thres_from_gmm`
    thre = thres_from_gmm(img)
    assert thre == thre_predicted


def test_fast_marching_seg():
    # create an image comprised of repeated 1D Gaussian distribution with mean value at 50th pixel and standard deviation of 2 pixels
    Gx = np.array([])
    for x in range(0, 101):
        Gx = np.insert(Gx, x, np.exp(-((x - 50) ** 2) / (2 * (2 ** 2))))

    img = np.repeat([Gx], repeats=30, axis=0)
    # place a seed in the region of mean value
    seed = (50, 15)
    # input default settings of the fast_marching_seg function
    stopping_value = 150
    sigma = 0.5
    # convert image format to comply with SimpleITK
    _, img_T1_255 = get_img_T1(img)
    # explicitly applying SimpleITK filters accordingly as in the fast_marching_seg function
    feature_img = sitk.GradientMagnitudeRecursiveGaussian(img_T1_255, sigma=sigma)
    speed_img = sitk.BoundedReciprocal(feature_img)
    fm_filter = sitk.FastMarchingBaseImageFilter()
    fm_filter.SetTrialPoints([seed])
    fm_filter.SetStoppingValue(stopping_value)
    fm_img = fm_filter.Execute(speed_img)
    fm_img = sitk.Cast(sitk.RescaleIntensity(fm_img), sitk.sitkUInt8)
    labels_predicted = sitk.GetArrayFromImage(fm_img)
    # the prediceted labels is obtained by explicitly running the fast_marching function
    labels_predicted = (~labels_predicted.astype(bool)).astype(int)
    # acquire labels by employing the fast_marching_seg function
    labels = fast_marching_seg(img, seed, sigma=sigma)
    np.testing.assert_array_equal(labels, labels_predicted)


def test_level_set_seg():
    # create an image comprised of repeated 1D Gaussian distribution with mean value at 50th pixel and standard deviation of 2 pixels
    Gx = np.array([])
    for x in range(0, 101):
        Gx = np.insert(Gx, x, np.exp(-((x - 50) ** 2) / (2 * (2 ** 2))))

    img = np.repeat([Gx], repeats=30, axis=0)
    # place a seed in the region of mean value
    seed = (50, 15)
    # input default settings of the fast_marching_seg function
    lower_threshold = None
    upper_threshold = None
    factor = 2
    max_rms_error = 0.02
    num_iter = 1000
    curvature_scaling = 0.5
    propagation_scaling = 1
    # convert image format to comply with SimpleITK
    _, img_T1_255 = get_img_T1(img)
    # explicitly applying SimpleITK filters accordingly and run default threshold algorithms as in the level_seg_set function
    seg = sitk.Image(img_T1_255.GetSize(), sitk.sitkUInt8)
    seg.CopyInformation(img_T1_255)
    seg[seed] = 1
    seg = sitk.BinaryDilate(seg, [1] * seg.GetDimension())
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
    # the prediceted labels is obtained by explicitly running the level_set_seg function
    labels_predicted = sitk.GetArrayFromImage(ls > 0)
    # acquire labels by employing level_set_seg function
    labels = level_set_seg(img, seed, lower_threshold=None, upper_threshold=None)
    np.testing.assert_array_equal(labels, labels_predicted)


def test_connected_threshold():
    # create an image with 4 layers of gray scales
    G1 = np.full((4, 4), 255)
    G2 = np.full((4, 4), 200)
    G3 = np.full((4, 4), 100)
    G4 = np.full((4, 4), 0)
    img = np.concatenate((G1, G2, G3, G4)).reshape(4, 4, 4)
    # seed at the first layer with the highest gray level and give a lower_threshold
    labels = connected_threshold(img, [(0, 0, 0)], lower_threshold=150)
    # because the gray levels are arranged in a sequantial order, gray level above the threshold should be all labeled 1, otherwise 0
    labels_predicted = np.concatenate(
        ((G1 / G1).astype(int), (G2 / G2).astype(int), G3 - G3, G4 - G4)
    ).reshape(4, 4, 4)
    np.testing.assert_array_equal(labels, labels_predicted)
    # seed at the first layer without giving a lower_threshold
    labels = connected_threshold(img, [(0, 0, 0)])
    # since no threshold is given, the default is to use thres_from_gmm to determine the threshold (200 in this case)
    labels_predicted = np.concatenate(
        ((G1 / G1).astype(int), (G2 / G2).astype(int), G3 - G3, G4 - G4)
    ).reshape(4, 4, 4)
    np.testing.assert_array_equal(labels, labels_predicted)


def test_confidence_connected_threshold():
    # create a data set featured with Gaussian distribution
    Good = False
    # to ensure the random number does not fall outside of 0 to 255 range (dynamic range of an 8-bit image)
    while Good == False:
        G1 = np.round(np.random.normal(loc=125, scale=25, size=(62500, 1)))
        if min(G1) > 0 and max(G1) < 255:
            Good = True

    # the data is distributed in the image by the order of each pixel's intensity
    img = np.sort(G1).reshape(250, 250).astype(int)
    # if we set multiplier to be 2.5, we are expected to connect around 99% of the pixels
    labels = confidence_connected_threshold(
        img, [(124, 127)], multiplier=2.5, num_iter=180
    )
    assert sum(sum(labels.astype(float))) / 62500 > 0.98


def test_neighborhood_connected_threshold():
    # define an image with 2D Gaussian distribution
    Gx = np.array([])
    for x in range(0, 7):
        Gx = np.insert(Gx, x, np.exp(-((x - 3) ** 2) / (2 * (3 ** 2))))

    Gxy = np.zeros((7, 7))

    for x in range(0, 7):
        for y in range(0, 7):
            Gxy[x, y] = (Gx[x]) * (Gx[y])

    img = 255 * (Gxy / Gxy.max())
    # pick the central pixel as the seed and threshold at 200
    lower_threshold = 200
    seed = (3, 3)
    labels = neighborhood_connected_threshold(
        img, [seed], lower_threshold=lower_threshold
    )
    # all the neighbor pixels within radius=(1,1,1) should fit in the threshold
    labels_predicted = np.zeros((7, 7))
    labels_predicted[3, 3] = 1
    np.testing.assert_array_equal(labels, labels_predicted)


def test_otsu():
    G1 = np.append(
        np.round(np.random.normal(loc=40, scale=10, size=(499, 1))), np.array([0])
    )
    G2 = np.append(
        np.round(np.random.normal(loc=220, scale=10, size=(499, 1))), np.array([255])
    )
    img = np.concatenate((G1, G2)).reshape((10, 10, 10))
    # seed inside
    labels = otsu(img, (9, 1, 6))
    labels_predicted = np.concatenate(
        ((G1 - G1).astype(int), (G2 / G2).astype(int))
    ).reshape((10, 10, 10))
    np.testing.assert_array_equal(labels, labels_predicted)
    # seed outside
    labels = otsu(img, (1, 1, 3))
    labels_predicted = np.concatenate(
        ((G1 + (255 - G1)).astype(int), (G2 - G2).astype(int))
    ).reshape((10, 10, 10))
    np.testing.assert_array_equal(labels, labels_predicted)


def test_gmm_seg():
    # define two groups of Gaussian distribution points with distinct mean values
    G1 = np.append(
        np.round(np.random.normal(loc=40, scale=10, size=(499, 1))), np.array([0])
    )
    G2 = np.append(
        np.round(np.random.normal(loc=220, scale=10, size=(499, 1))), np.array([255])
    )
    # construct a 3D image with the two groups of points
    img = np.concatenate((G1, G2)).reshape((10, 10, 10))
    # 1s should be labeled to the positions where G2 population are located
    labels_predicted = np.concatenate(
        ((G1 - G1).astype(int), (G2 / G2).astype(int))
    ).reshape((10, 10, 10))
    # the seed is located at a randomly selected point within G2 distribution
    labels = gmm_seg(img, (9, 9, 6))
    np.testing.assert_array_equal(labels, labels_predicted)
