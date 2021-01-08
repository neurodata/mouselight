import pytest
from brainlit.utils.session import NeuroglancerSession
from brainlit.utils.upload import upload_volumes, upload_segments
from brainlit.feature_extraction import neighborhood as nbrhood
import numpy as np
import pandas as pd
from cloudvolume import CloudVolume
import cloudvolume
from cloudvolume.lib import Bbox
import glob
import os
from pathlib import Path


top_level = Path(__file__).parents[1] / "data"
input = (top_level / "data_octree").as_posix()
url = (top_level / "test_upload").as_uri()
url_seg = url + "_segments"
url = url + "/serial"

SIZE = 2
SEGLIST = [2]
OFF = [15, 15, 15]

##############
### inputs ###
##############


def test_init_bad_inputs():
    """Tests that proper errors are raised when bad inputs are given to __init__ method."""
    with pytest.raises(TypeError):
        nbrhood.NeighborhoodFeatures(url=0, radius=SIZE, offset=OFF)
    with pytest.raises(NotImplementedError):
        nbrhood.NeighborhoodFeatures(url="asdf", radius=SIZE, offset=OFF)
    with pytest.raises(TypeError):
        nbrhood.NeighborhoodFeatures(url=url, radius=0.5, offset=OFF)
    with pytest.raises(ValueError):
        nbrhood.NeighborhoodFeatures(
            url=url, radius=-1, offset=OFF, segment_url=url_seg
        )
    with pytest.raises(TypeError):
        nbrhood.NeighborhoodFeatures(
            url=url, radius=SIZE, offset=12, segment_url=url_seg
        )
    with pytest.raises(TypeError):
        nbrhood.NeighborhoodFeatures(url=url, radius=SIZE, offset=OFF, segment_url=0)
    with pytest.raises(NotImplementedError):
        nbrhood.NeighborhoodFeatures(
            url=url, radius=SIZE, offset=OFF, segment_url="asdf"
        )


def test_fit_bad_inputs():
    """Tests that proper errors are raised when bad inputs are given to fit method."""
    nbr = nbrhood.NeighborhoodFeatures(
        url=url, radius=SIZE, offset=[15, 15, 15], segment_url=url_seg
    )
    with pytest.raises(TypeError):
        nbr.fit(seg_ids=1, num_verts=5, file_path="demo", batch_size=1000)
    with pytest.raises(cloudvolume.exceptions.SkeletonDecodeError):
        nbr.fit(seg_ids=[1], num_verts=5, file_path="demo", batch_size=1000)


##################
### validation ###
##################


def test_neighborhood():
    """Tests that neighborhood data is generated correctly."""
    nbr = nbrhood.NeighborhoodFeatures(
        url=url, radius=1, offset=[15, 15, 15], segment_url=url_seg
    )
    df_nbr = nbr.fit([2], 5)
    assert df_nbr.shape == (10, 30)  # 5on, 5off for each swc
    assert not df_nbr.empty


# def test_linear_features():
#     lin = linear_features.LinearFeatures(url=URL, radius=1, offset=[15, 15, 15], segment_url=URL+"_segments")
#     lin.add_filter("gaussian", sigma=[1, 1, 0.3])
#     df_lin = lin.fit([2, 7], 5)
#     assert df_lin.shape == (20, 4)
#     assert not df_lin.empty

#     df_lin = lin.fit([2, 7], 5, include_neighborhood=True)
#     assert df_lin.shape == (20, 31)
#     assert not df_lin.empty


# def test_add_filter():
#     lin = linear_features.LinearFeatures(url=URL, radius=1, offset=[15, 15, 15], segment_url=URL+"_segments")
#     with pytest.raises(ValueError):
#         lin.add_filter("asdf")
#     lin.add_filter("gaussian gradient", sigma=[1, 1, 0.3])
#     lin.add_filter("gaussian laplace", sigma=[1, 1, 0.3])
#     lin.add_filter("gabor", sigma=[1, 1, 0.3], phi=[0, 0], frequency=2)
#     lin.add_filter("gabor", sigma=[1, 1, 0.3], phi=[0, np.pi / 2], frequency=2)
#     df_lin = lin.fit([2, 7], 5)
#     assert df_lin.shape == (20, 7)


def test_file_write():
    """Tests that files are written correctly."""
    files = sorted(glob.glob("*.feather"))
    for f in sorted(files):
        os.remove(f)

    nbr = nbrhood.NeighborhoodFeatures(
        url=url, radius=1, offset=[15, 15, 15], segment_url=url_seg
    )
    nbr.fit([2], 5, file_path="test", batch_size=10)

    files = sorted(glob.glob("*.feather"))
    for f in sorted(files):
        print(f)
        os.remove(f)
    assert files == ["test0_10_2_4.feather"]

    df_nbr = nbr.fit([2], 5, file_path="test", batch_size=10, start_seg=2, start_vert=0)
    files = sorted(glob.glob("*.feather"))
    for f in sorted(files):
        os.remove(f)
    assert files == ["test0_10_2_4.feather"]


# def test_parallel():
#     files = sorted(glob.glob("*.feather"))
#     for f in sorted(files):
#         os.remove(f)

#     nbr = nbrhood.NeighborhoodFeatures(
#         url=URL, radius=1, offset=[15, 15, 15], segment_url=URL + "_segments"
#     )
#     nbr.fit(seg_ids=[2, 7], num_verts=4, file_path="test", batch_size=4, n_jobs=2)
#     files = sorted(glob.glob("*.feather"))
#     for f in sorted(files):
#         os.remove(f)
#     assert files == [
#         "test0_4_2_1.feather",
#         "test0_4_7_1.feather",
#         "test4_8_2_3.feather",
#         "test4_8_7_3.feather",
#     ]

#     nbr = nbrhood.NeighborhoodFeatures(
#         url=URL, radius=1, offset=[15, 15, 15], segment_url=URL + "_segments"
#     )
#     nbr.fit(seg_ids=[2, 7], num_verts=2, file_path="test", batch_size=6, n_jobs=2)
#     files = sorted(glob.glob("*.feather"))
#     for f in sorted(files):
#         os.remove(f)
#     assert files == ["test0_4_2_1.feather", "test0_4_7_1.feather"]
