import pytest
import numpy as np
from brainlit.preprocessing.image_process import (
    gabor_filter,
    getLargestCC,
    removeSmallCCs,
    griddedInterpolant,
)
from numpy.testing import (
    assert_equal,
    assert_allclose,
    assert_array_equal,
    assert_almost_equal,
    assert_array_almost_equal,
)


####################
### input checks ###
####################


def test_gabor_filter_bad_inputs():
    """Tests that errors are raised when bad inputs are given to image_process.gabor_filter."""
    with pytest.raises(TypeError):
        gabor_filter(input=0, sigma=0, phi=0, frequency=5)
    with pytest.raises(TypeError):
        gabor_filter(input=np.ones((5, 5)), sigma="a", phi=0, frequency=5)
    with pytest.raises(TypeError):
        gabor_filter(input=np.ones((5, 5)), sigma=1, phi="a", frequency=5)
    with pytest.raises(RuntimeError):
        gabor_filter(input=np.ones((5, 5)), sigma=[1, 1, 1], phi=0, frequency=5)
    with pytest.raises(RuntimeError):
        gabor_filter(input=np.ones((5, 5)), sigma=[1, 1], phi=[1, 1], frequency=5)
    with pytest.raises(RuntimeError):
        gabor_filter(input=np.ones((5, 5)), sigma=1, phi=1, frequency=1, output="a")
    with pytest.raises(RuntimeError):
        gabor_filter(
            input=np.ones((5, 5)), sigma=1, phi=1, frequency=1, output=np.ones((3, 3))
        )
    with pytest.raises(RuntimeError):
        gabor_filter(
            input=np.ones((5, 5)), sigma=1, phi=1, frequency=1, mode="nonsensical"
        )
    with pytest.raises(TypeError):
        gabor_filter(input=np.ones((5, 5)), sigma=1, phi=1, frequency=1, cval="a")
    with pytest.raises(TypeError):
        gabor_filter(input=np.ones((5, 5)), sigma=1, phi=1, frequency=1, truncate="a")


def test_gabor_filter_valid_inputs():
    """Tests that no errors are raised when valid inputs are given to image_process.gabor_filter."""
    gabor_filter(input=np.ones((5, 5)), sigma=1, phi=1, frequency=1)
    gabor_filter(input=np.ones((5, 5)), sigma=1.0, phi=1, frequency=1)
    gabor_filter(
        input=np.ones((5, 5)), sigma=np.dtype("float32").type(1), phi=1, frequency=1
    )

    gabor_filter(input=np.ones((5, 5)), sigma=[1, 1], phi=1, frequency=1)
    gabor_filter(input=np.ones((5, 5)), sigma=[1.0, 1.0], phi=1, frequency=1)
    gabor_filter(
        input=np.ones((5, 5)), sigma=np.ones(2).astype("float32"), phi=1, frequency=1
    )

    gabor_filter(input=np.ones((5, 5)), sigma=1, phi=1, frequency=1)
    gabor_filter(input=np.ones((5, 5)), sigma=1, phi=1.0, frequency=1)
    gabor_filter(
        input=np.ones((5, 5)), sigma=1, phi=np.dtype("float32").type(1), frequency=1
    )

    gabor_filter(input=np.ones((5, 5)), sigma=1, phi=[1], frequency=1)
    gabor_filter(input=np.ones((5, 5)), sigma=1, phi=[1.0], frequency=1)
    gabor_filter(
        input=np.ones((5, 5)), sigma=1, phi=np.ones(1).astype("float32"), frequency=1
    )

    gabor_filter(input=np.ones((5, 5)), sigma=1, phi=1, frequency=1)
    gabor_filter(input=np.ones((5, 5)), sigma=1, phi=1, frequency=1.0)
    gabor_filter(
        input=np.ones((5, 5)), sigma=1, phi=1, frequency=np.dtype("float32").type(1)
    )

    gabor_filter(input=np.ones((5, 5)), sigma=1, phi=1, frequency=1, offset=1)
    gabor_filter(input=np.ones((5, 5)), sigma=1, phi=1, frequency=1, offset=1.0)
    gabor_filter(
        input=np.ones((5, 5)),
        sigma=1,
        phi=1,
        frequency=1,
        offset=np.dtype("float32").type(1),
    )

    gabor_filter(
        input=np.ones((5, 5)), sigma=1, phi=1, frequency=1, output=np.ones((5, 5))
    )
    gabor_filter(input=np.ones((5, 5)), sigma=1, phi=1, frequency=1, output="float32")

    gabor_filter(input=np.ones((5, 5)), sigma=1, phi=1, frequency=1, cval=1)
    gabor_filter(input=np.ones((5, 5)), sigma=1, phi=1, frequency=1, cval=1.0)
    gabor_filter(
        input=np.ones((5, 5)),
        sigma=1,
        phi=1,
        frequency=1,
        cval=np.dtype("float32").type(1),
    )

    gabor_filter(input=np.ones((5, 5)), sigma=1, phi=1, frequency=1, truncate=1)
    gabor_filter(input=np.ones((5, 5)), sigma=1, phi=1, frequency=1, truncate=1.0)
    gabor_filter(
        input=np.ones((5, 5)),
        sigma=1,
        phi=1,
        frequency=1,
        truncate=np.dtype("float32").type(1),
    )


def test_getLargestCC_bad_input():
    with pytest.raises(TypeError):
        getLargestCC(1)
    with pytest.raises(ValueError):
        getLargestCC(np.zeros((5, 5)))


def test_getLargestCC_valid_input():
    getLargestCC(np.ones((5, 5)))


def test_removeSmallCCs_bad_input():
    with pytest.raises(TypeError):
        removeSmallCCs(1, 2)
    with pytest.raises(TypeError):
        removeSmallCCs(np.ones((5, 5)), [2])
    with pytest.raises(ValueError):
        removeSmallCCs(np.zeros((5, 5)), 2)


def test_removeSmallCCs_valid_input():
    removeSmallCCs(np.ones((5, 5)), 2)


############################
### functionality checks ###
############################


def test_gaussian_truncate():
    """Tests that Gaussian filters can be truncated at different widths."""
    arr = np.zeros((100, 100), float)
    arr[50, 50] = 1
    num_nonzeros_2 = (np.abs(gabor_filter(arr, 5, 0, 5, truncate=2)[0]) > 0).sum()
    assert_equal(num_nonzeros_2, 21 ** 2)
    num_nonzeros_5 = (np.abs(gabor_filter(arr, 5, 0, 5, truncate=5)[0]) > 0).sum()
    assert_equal(num_nonzeros_5, 51 ** 2)

    f = np.abs(gabor_filter(arr, [0.5, 2.5], 0, 5, truncate=3.5)[0])
    fpos = f > 0
    n0 = fpos.any(axis=0).sum()
    # n0 should be 2*int(2.5*3.5 + 0.5) + 1
    assert_equal(n0, 19)
    n1 = fpos.any(axis=1).sum()
    # n1 should be 2*int(0.5*3.5 + 0.5) + 1
    assert_equal(n1, 5)


def test_gabor_filter():
    """Tests that the gabor filter works correctly."""
    # Test generic case
    input = np.array([[1, 2, 3], [2, 4, 6]], np.float32)
    real, imag = gabor_filter(input, 1.0, 0, 5)
    assert_equal(input.dtype, real.dtype)
    assert_equal(input.shape, real.shape)
    assert_equal(input.dtype, imag.dtype)
    assert_equal(input.shape, imag.shape)

    # Test using otype
    input = np.arange(100 * 100).astype(np.float32)
    input.shape = (100, 100)
    otype = np.float64
    real, imag = gabor_filter(input, [1.0, 1.0], 0, 5, output=otype)
    assert_equal(real.dtype.type, np.float64)
    assert_equal(input.shape, real.shape)
    assert_equal(imag.dtype.type, np.float64)
    assert_equal(input.shape, imag.shape)

    # Tests that inputting an output array works
    input = np.arange(100 * 100).astype(np.float32)
    input.shape = (100, 100)
    output = np.zeros([100, 100]).astype(np.float32)
    gabor_filter(input, [1.0, 1.0], 0, 5, output=output)
    assert_equal(input.dtype, output.dtype)
    assert_equal(input.shape, output.shape)

    # Tests that list of sigmas is same as single
    input = np.arange(100 * 100).astype(np.float32)
    input.shape = (100, 100)
    otype = np.float64
    real1, imag1 = gabor_filter(input, [1.0, 1.0], 0, 5, output=otype)
    real2, imag2 = gabor_filter(input, 1.0, 0, 5, output=otype)
    assert_array_almost_equal(real1, real2)
    assert_array_almost_equal(imag1, imag2)

    # Tests that list of phi works
    input = np.arange(100 * 100 * 100).astype(np.float32)
    input.shape = (100, 100, 100)
    otype = np.float64
    real1, imag1 = gabor_filter(input, 1.0, [0.5, 0.5], 5, output=otype)
    real2, imag2 = gabor_filter(input, 1.0, 0.5, 5, output=otype)
    assert_array_almost_equal(real1, real2)
    assert_array_almost_equal(imag1, imag2)


def test_getLargestCC():
    """Tests that getLargestCC works correctly."""
    img = np.zeros([50, 50])
    img[25, 25] = 1
    output = getLargestCC(img)
    assert_array_equal(img, output)

    img[25, 26] = 1
    img[0, 0] = 1
    expected_output = img
    expected_output[0, 0] = 0

    output = getLargestCC(img)
    assert_array_equal(expected_output, output)


def test_removeSmallCCs():
    """Tests that removeSmallCCs works correctly."""
    img = np.zeros([50, 50])
    img[25, 25] = 1
    img[25, 26] = 1
    output = removeSmallCCs(img, 1)
    assert_array_equal(img, output)

    img = np.zeros([50, 50])
    expected_output = np.zeros([50, 50])
    for i in range(0, 2):
        for j in range(0, 2):
            img[i, j] = 1
    for i in range(4, 7):
        for j in range(4, 7):
            img[i, j] = 1
            expected_output[i, j] = 1
    for i in range(20, 30):
        for j in range(20, 30):
            img[i, j] = 1
            expected_output[i, j] = 1
    output = removeSmallCCs(img, 5).astype(int)
    assert_array_equal(expected_output, output)
