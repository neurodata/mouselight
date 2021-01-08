# import pytest
# import numpy as np
# from brainlit import preprocessing as pp
# from numpy.testing import (
#     assert_equal,
#     assert_allclose,
#     assert_array_equal,
#     assert_almost_equal,
#     assert_array_almost_equal,
# )


# def test_gaussian_truncate():
#     # Test that Gaussian filters can be truncated at different widths.
#     # These tests only check that the result has the expected number
#     # of nonzero elements.
#     arr = np.zeros((100, 100), float)
#     arr[50, 50] = 1
#     num_nonzeros_2 = (np.abs(pp.gabor_filter(arr, 5, 0, 5, truncate=2)[0]) > 0).sum()
#     assert_equal(num_nonzeros_2, 21 ** 2)
#     num_nonzeros_5 = (np.abs(pp.gabor_filter(arr, 5, 0, 5, truncate=5)[0]) > 0).sum()
#     assert_equal(num_nonzeros_5, 51 ** 2)

#     f = np.abs(pp.gabor_filter(arr, [0.5, 2.5], 0, 5, truncate=3.5)[0])
#     fpos = f > 0
#     n0 = fpos.any(axis=0).sum()
#     # n0 should be 2*int(2.5*3.5 + 0.5) + 1
#     assert_equal(n0, 19)
#     n1 = fpos.any(axis=1).sum()
#     # n1 should be 2*int(0.5*3.5 + 0.5) + 1
#     assert_equal(n1, 5)


# def test_gabor_filter():
#     # Test generic case
#     input = np.array([[1, 2, 3], [2, 4, 6]], np.float32)
#     real, imag = pp.gabor_filter(input, 1.0, 0, 5)
#     assert_equal(input.dtype, real.dtype)
#     assert_equal(input.shape, real.shape)
#     assert_equal(input.dtype, imag.dtype)
#     assert_equal(input.shape, imag.shape)

#     # Test using otype
#     input = np.arange(100 * 100).astype(np.float32)
#     input.shape = (100, 100)
#     otype = np.float64
#     real, imag = pp.gabor_filter(input, [1.0, 1.0], 0, 5, output=otype)
#     assert_equal(real.dtype.type, np.float64)
#     assert_equal(input.shape, real.shape)
#     assert_equal(imag.dtype.type, np.float64)
#     assert_equal(input.shape, imag.shape)

#     # Tests that inputting an output array works
#     input = np.arange(100 * 100).astype(np.float32)
#     input.shape = (100, 100)
#     output = np.zeros([100, 100]).astype(np.float32)
#     pp.gabor_filter(input, [1.0, 1.0], 0, 5, output=output)
#     assert_equal(input.dtype, output.dtype)
#     assert_equal(input.shape, output.shape)

#     # Tests that list of sigmas is same as single
#     input = np.arange(100 * 100).astype(np.float32)
#     input.shape = (100, 100)
#     otype = np.float64
#     real1, imag1 = pp.gabor_filter(input, [1.0, 1.0], 0, 5, output=otype)
#     real2, imag2 = pp.gabor_filter(input, 1.0, 0, 5, output=otype)
#     assert_array_almost_equal(real1, real2)
#     assert_array_almost_equal(imag1, imag2)

#     # Tests that list of phi works
#     input = np.arange(100 * 100 * 100).astype(np.float32)
#     input.shape = (100, 100, 100)
#     otype = np.float64
#     real1, imag1 = pp.gabor_filter(input, 1.0, [0.5, 0.5], 5, output=otype)
#     real2, imag2 = pp.gabor_filter(input, 1.0, 0.5, 5, output=otype)
#     assert_array_almost_equal(real1, real2)
#     assert_array_almost_equal(imag1, imag2)


# def test_getLargestCC():
#     img = np.zeros([50, 50])
#     with pytest.raises(ValueError):
#         pp.getLargestCC(img)
#     img[25, 25] = 1
#     output = pp.getLargestCC(img)
#     assert_array_equal(img, output)

#     img[25, 26] = 1
#     img[0, 0] = 1
#     expected_output = img
#     expected_output[0, 0] = 0

#     output = pp.getLargestCC(img)
#     assert_array_equal(expected_output, output)


# # removeSmallCCs(segmentation, size)
# def test_removeSmallCCs():
#     img = np.zeros([50, 50])
#     with pytest.raises(ValueError):
#         pp.getLargestCC(img)
#     img[25, 25] = 1
#     img[25, 26] = 1
#     output = pp.removeSmallCCs(img, 1)
#     assert_array_equal(img, output)

#     img = np.zeros([50, 50])
#     expected_output = np.zeros([50, 50])
#     for i in range(0, 2):
#         for j in range(0, 2):
#             img[i, j] = 1
#     for i in range(4, 7):
#         for j in range(4, 7):
#             img[i, j] = 1
#             expected_output[i, j] = 1
#     for i in range(20, 30):
#         for j in range(20, 30):
#             img[i, j] = 1
#             expected_output[i, j] = 1
#     output = pp.removeSmallCCs(img, 5).astype(int)
#     assert_array_equal(expected_output, output)
