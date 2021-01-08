# import pytest
# import numpy as np
# from brainlit import preprocessing
# from numpy.testing import (
#     assert_equal,
#     assert_allclose,
#     assert_array_equal,
#     assert_almost_equal,
#     assert_array_almost_equal,
# )


# def test_center():
#     img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#     centered_image = np.array([[-4, -3, -2], [-1, 0, 1], [2, 3, 4]])
#     assert_array_equal(preprocessing.center(img), centered_image)


# def test_contrast_normalize():
#     img = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])
#     expected = np.array([[0, 2.12132034, 0], [0, 0, 0], [0, -2.12132034, 0]])
#     assert_almost_equal(preprocessing.contrast_normalize(img), expected)


# def test_pad_undopad_transform():
#     np.random.seed(6)
#     img = np.random.randint(0, 256, size=(50, 50))
#     window_size = np.array([5, 5])
#     step_size = np.array([2, 2])
#     padded, pad_size = preprocessing.window_pad(img, window_size, step_size)
#     new_img = preprocessing.undo_pad(padded, pad_size)
#     assert_array_equal(img, new_img)


# def test_pad_undopad_transform_3D():
#     np.random.seed(6)
#     img = np.random.randint(0, 256, size=(50, 50, 50))
#     window_size = np.array([5, 5, 5])
#     step_size = np.array([2, 2, 2])
#     padded, pad_size = preprocessing.window_pad(img, window_size, step_size)
#     new_img = preprocessing.undo_pad(padded, pad_size)
#     assert_array_equal(img, new_img)


# def test_image_vector_transform():
#     np.random.seed(6)

#     img = np.random.randint(0, 256, size=(10, 10))
#     window_size = np.array([3, 3])
#     step_size = np.array([1, 1])
#     vector = preprocessing.vectorize_img(img, window_size, step_size)
#     new_image = preprocessing.imagize_vector(vector, img.shape, window_size, step_size)
#     assert_array_equal(img, new_image)

#     img = np.random.randint(0, 256, size=(20, 20))
#     window_size = np.array([3, 3])
#     step_size = np.array([1, 1])
#     vector = preprocessing.vectorize_img(img, window_size, step_size)
#     new_image = preprocessing.imagize_vector(vector, img.shape, window_size, step_size)
#     assert_array_equal(img, new_image)


# def test_image_vector_transform_3D():
#     np.random.seed(6)

#     img = np.random.randint(0, 256, size=(10, 10, 10))
#     window_size = np.array([3, 3, 3])
#     step_size = np.array([1, 1, 1])
#     vector = preprocessing.vectorize_img(img, window_size, step_size)
#     new_image = preprocessing.imagize_vector(vector, img.shape, window_size, step_size)
#     assert_array_equal(img, new_image)

#     img = np.random.randint(0, 256, size=(20, 20, 20))
#     window_size = np.array([3, 3, 3])
#     step_size = np.array([1, 1, 1])
#     vector = preprocessing.vectorize_img(img, window_size, step_size)
#     new_image = preprocessing.imagize_vector(vector, img.shape, window_size, step_size)
#     assert_array_equal(img, new_image)


# def test_window_pad_2D():
#     # Trivial example
#     img = np.zeros([50, 50])
#     window_size = np.array([3, 3])
#     step_size = np.array([1, 1])

#     [img_padded, padding] = preprocessing.window_pad(img, window_size, step_size)
#     assert img_padded.shape == (52, 52)
#     assert_array_equal(padding, np.array([[1, 1], [1, 1]]))

#     img = np.zeros([50, 50])
#     window_size = np.array([5, 5])
#     step_size = np.array([1, 1])

#     [img_padded, padding] = preprocessing.window_pad(img, window_size, step_size)
#     assert img_padded.shape == (54, 54)
#     assert_array_equal(padding, np.array([[2, 2], [2, 2]]))

#     img = np.zeros([50, 50])
#     window_size = np.array([5, 5])
#     step_size = np.array([1, 4])

#     [img_padded, padding] = preprocessing.window_pad(img, window_size, step_size)
#     assert img_padded.shape == (54, 52)
#     assert_array_equal(padding, np.array([[2, 2], [2, 0]]))

#     img = np.zeros([50, 50])
#     window_size = np.array([3])
#     step_size = np.array([1, 1])
#     with pytest.raises(ValueError):
#         preprocessing.window_pad(img, window_size, step_size)
#     window_size = np.array([3, 3, 3])
#     with pytest.raises(ValueError):
#         preprocessing.window_pad(img, window_size, step_size)
#     window_size = np.array([[3, 3], [3, 3]])
#     with pytest.raises(ValueError):
#         preprocessing.window_pad(img, window_size, step_size)

#     window_size = np.array([3, 3])
#     step_size = np.array([1])
#     with pytest.raises(ValueError):
#         preprocessing.window_pad(img, window_size, step_size)
#     step_size = np.array([1, 1, 1])
#     with pytest.raises(ValueError):
#         preprocessing.window_pad(img, window_size, step_size)
#     step_size = np.array([[1, 1], [1, 1]])
#     with pytest.raises(ValueError):
#         preprocessing.window_pad(img, window_size, step_size)


# def test_window_pad_3D():
#     img = np.zeros([50, 50, 50])
#     window_size = np.array([3, 3, 3])
#     step_size = np.array([1, 1, 1])
#     [img_padded, padding] = preprocessing.window_pad(img, window_size, step_size)
#     assert img_padded.shape == (52, 52, 52)
#     assert_array_equal(padding, np.array([[1, 1], [1, 1], [1, 1]]))

#     img = np.zeros([50, 50, 50])
#     window_size = np.array([5, 5, 5])
#     step_size = np.array([1, 1, 1])
#     [img_padded, padding] = preprocessing.window_pad(img, window_size, step_size)
#     assert img_padded.shape == (54, 54, 54)
#     assert_array_equal(padding, np.array([[2, 2], [2, 2], [2, 2]]))

#     img = np.zeros([50, 50, 50])
#     window_size = np.array([5, 5, 5])
#     step_size = np.array([1, 4, 1])
#     [img_padded, padding] = preprocessing.window_pad(img, window_size, step_size)
#     assert img_padded.shape == (54, 52, 54)
#     assert_array_equal(padding, np.array([[2, 2], [2, 0], [2, 2]]))


# def test_vectorize_image():
#     np.random.seed(6)

#     img = np.random.randint(0, 256, size=(10, 10))
#     window_size = np.array([3, 3])
#     step_size = np.array([1, 1])
#     vector = preprocessing.vectorize_img(img, window_size, step_size)
#     assert_array_equal(vector[:, 0].flatten(), img[0:3, 0:3].flatten())
#     assert_array_equal(vector[:, 5].flatten(), img[0:3, 5:8].flatten())
#     assert_array_equal(vector[:, 8].flatten(), img[1:4, 0:3].flatten())

#     img = np.random.randint(0, 256, size=(10, 10))
#     window_size = np.array([3, 3])
#     step_size = np.array([2, 1])
#     vector = preprocessing.vectorize_img(img, window_size, step_size)
#     assert_array_equal(vector[:, 0].flatten(), img[0:3, 0:3].flatten())
#     assert_array_equal(vector[:, 5].flatten(), img[0:3, 5:8].flatten())
#     assert_array_equal(vector[:, 8].flatten(), img[2:5, 0:3].flatten())

#     img = np.zeros([50, 50])
#     window_size = np.array([3])
#     step_size = np.array([1, 1])
#     with pytest.raises(ValueError):
#         preprocessing.vectorize_img(img, window_size, step_size)
#     window_size = np.array([3, 3, 3])
#     with pytest.raises(ValueError):
#         preprocessing.vectorize_img(img, window_size, step_size)
#     window_size = np.array([[3, 3], [3, 3]])
#     with pytest.raises(ValueError):
#         preprocessing.vectorize_img(img, window_size, step_size)

#     window_size = np.array([3, 3])
#     step_size = np.array([1])
#     with pytest.raises(ValueError):
#         preprocessing.vectorize_img(img, window_size, step_size)
#     step_size = np.array([1, 1, 1])
#     with pytest.raises(ValueError):
#         preprocessing.vectorize_img(img, window_size, step_size)
#     step_size = np.array([[1, 1], [1, 1]])
#     with pytest.raises(ValueError):
#         preprocessing.vectorize_img(img, window_size, step_size)


# def test_vectorize_image_3D():
#     np.random.seed(6)

#     img = np.random.randint(0, 256, size=(10, 10, 10))
#     window_size = np.array([3, 3, 3])
#     step_size = np.array([1, 1, 1])
#     vector = preprocessing.vectorize_img(img, window_size, step_size)
#     assert_array_equal(vector[:, 0].flatten(), img[0:3, 0:3, 0:3].flatten())
#     assert_array_equal(vector[:, 5].flatten(), img[0:3, 0:3, 5:8].flatten())
#     assert_array_equal(vector[:, 8].flatten(), img[0:3, 1:4, 0:3].flatten())

#     img = np.random.randint(0, 256, size=(10, 10, 10))
#     window_size = np.array([3, 3, 3])
#     step_size = np.array([1, 2, 1])
#     vector = preprocessing.vectorize_img(img, window_size, step_size)
#     assert_array_equal(vector[:, 0].flatten(), img[0:3, 0:3, 0:3].flatten())
#     assert_array_equal(vector[:, 5].flatten(), img[0:3, 0:3, 5:8].flatten())
#     assert_array_equal(vector[:, 8].flatten(), img[0:3, 2:5, 0:3].flatten())


# def test_undo_pad():
#     np.random.seed(6)
#     img = np.random.randint(0, 256, size=(10, 10))
#     padding = np.array([2, 2])
#     with pytest.raises(ValueError):
#         preprocessing.undo_pad(img, padding)

#     padding = np.array([[2, 2], [2, 2], [2, 2]])
#     with pytest.raises(ValueError):
#         preprocessing.undo_pad(img, padding)


# def test_imagize_vector():
#     img = np.zeros([50, 50])
#     orig_shape = np.array([50, 50])
#     window_size = np.array([3])
#     step_size = np.array([1, 1])
#     with pytest.raises(ValueError):
#         preprocessing.imagize_vector(img, orig_shape, window_size, step_size)
#     window_size = np.array([3, 3, 3])
#     with pytest.raises(ValueError):
#         preprocessing.imagize_vector(img, orig_shape, window_size, step_size)
#     window_size = np.array([[3, 3], [3, 3]])
#     with pytest.raises(ValueError):
#         preprocessing.imagize_vector(img, orig_shape, window_size, step_size)

#     window_size = np.array([3, 3])
#     step_size = np.array([1])
#     with pytest.raises(ValueError):
#         preprocessing.imagize_vector(img, orig_shape, window_size, step_size)
#     step_size = np.array([1, 1, 1])
#     with pytest.raises(ValueError):
#         preprocessing.imagize_vector(img, orig_shape, window_size, step_size)
#     step_size = np.array([[1, 1], [1, 1]])
#     with pytest.raises(ValueError):
#         preprocessing.imagize_vector(img, orig_shape, window_size, step_size)


# def test_whiten():
#     img = np.zeros([50, 50])
#     window_size = np.array([3])
#     step_size = np.array([1, 1])
#     with pytest.raises(ValueError):
#         preprocessing.whiten(img, window_size, step_size)
#     window_size = np.array([3, 3, 3])
#     with pytest.raises(ValueError):
#         preprocessing.whiten(img, window_size, step_size)
#     window_size = np.array([[3, 3], [3, 3]])
#     with pytest.raises(ValueError):
#         preprocessing.whiten(img, window_size, step_size)

#     window_size = np.array([3, 3])
#     step_size = np.array([1])
#     with pytest.raises(ValueError):
#         preprocessing.whiten(img, window_size, step_size)
#     step_size = np.array([1, 1, 1])
#     with pytest.raises(ValueError):
#         preprocessing.whiten(img, window_size, step_size)
#     step_size = np.array([[1, 1], [1, 1]])
#     with pytest.raises(ValueError):
#         preprocessing.whiten(img, window_size, step_size)

#     window_size = np.array([3, 3])
#     step_size = np.array([1, 1])
#     with pytest.raises(ValueError):
#         preprocessing.whiten(img, window_size, step_size, type="as")
