import pytest
from brainlit.utils.session import NeuroglancerSession
from brainlit.feature_extraction import neighborhood as nbrhood
from brainlit.utils.upload import upload_volumes, upload_segments
import numpy as np
import pandas as pd
from cloudvolume import CloudVolume
from pathlib import Path

top_level = Path(__file__).parents[1] / "data"
input = (top_level / "data_octree").as_posix()
url = (top_level / "test_upload").as_uri()
url_seg = url + "_segments"
url = url + "/serial"

SIZE = 2
OFF = [15, 15, 15]


@pytest.fixture
def gen_array():
    nbr = nbrhood.NeighborhoodFeatures(
        url=url, radius=SIZE, offset=[15, 15, 15], segment_url=url_seg
    )
    df_nbr = nbr.fit([2], 5)
    ind = SIZE * 2 + 1
    arr = df_nbr.iloc[2, 3:].values.reshape((ind, ind, ind))
    return arr


##############
### inputs ###
##############


def test_subneighborhood_bad_inputs(gen_array):
    arr = gen_array
    arr_flat = arr[0, :, :].flatten()
    arr_flat3 = arr.flatten()
    with pytest.raises(TypeError):
        nbrhood.subsample("asdf", (5, 5), (3, 3))
    # 2d
    with pytest.raises(TypeError):
        nbrhood.subsample(arr_flat, 0, (3, 3))
    with pytest.raises(ValueError):
        nbrhood.subsample(arr_flat, (5,), (3, 3))
    with pytest.raises(TypeError):
        nbrhood.subsample(arr_flat, (5, 5), ("a", "b"))
    with pytest.raises(ValueError):
        nbrhood.subsample(arr_flat, (5, 5), (3,))
    with pytest.raises(ValueError):
        nbrhood.subsample(arr_flat, (5, 5), (-1, -1))
    # 3d
    with pytest.raises(TypeError):
        nbrhood.subsample(arr_flat3, 0, (3, 3, 3))
    with pytest.raises(ValueError):
        nbrhood.subsample(arr_flat3, (5,), (3, 3, 3))
    with pytest.raises(TypeError):
        nbrhood.subsample(arr_flat3, (5, 5, 5), 0)
    with pytest.raises(ValueError):
        nbrhood.subsample(arr_flat3, (5, 5, 5), (3,))
    with pytest.raises(ValueError):
        nbrhood.subsample(arr_flat3, (5, 5, 5), (-1, -1, -1))


##################
### validation ###
##################


def test_2d(gen_array):
    arr = gen_array
    a1 = arr[2, :, :].flatten()
    sub_a1 = nbrhood.subsample(a1, (5, 5), (3, 3)).reshape((3, 3))
    assert np.array_equal(sub_a1, arr[2, 1:4, 1:4])


def test_even(gen_array):
    arr = gen_array
    a1 = arr[2, :, :].flatten()
    sub_a1_even = nbrhood.subsample(a1, (5, 5), (4, 4)).reshape((4, 4))
    assert np.array_equal(sub_a1_even, arr[2, 0:4, 0:4])


def test_3d(gen_array):
    arr = gen_array
    a2 = arr.flatten()
    sub_a2 = nbrhood.subsample(a2, (5, 5, 5), (3, 3, 3)).reshape((3, 3, 3))
    assert np.array_equal(sub_a2, arr[1:4, 1:4, 1:4])
