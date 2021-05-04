import os
import numpy as np
from brainlit.algorithms.detect_somas import find_somas
from pytest import raises
from pathlib import Path

dir = "s3://open-neurodata/brainlit/brain1"
dir_segments = "s3://open-neurodata/brainlit/brain1_segments"

cwd = Path(os.path.abspath(__file__))
root_dir = cwd.parents[4]
data_dir = os.path.join(root_dir, "data", "test_detect")

##############
### inputs ###
##############


def test_find_somas_bad_input():
    volume = []
    res = ""

    # test input volume is numpy.ndarray
    with raises(TypeError, match=r"should be <class 'numpy.ndarray'>"):
        find_somas(volume, res)
    volume = np.array([1, "a", 2])
    with raises(TypeError, match=r"elements should be <class 'numpy.uint16'>"):
        find_somas(volume, res)
    volume = np.zeros((2, 2), dtype=np.uint16)
    # test input volume must be three-dimensional
    with raises(ValueError, match=r"Input volume must be three-dimensional"):
        find_somas(volume, res)
    volume = np.zeros((19, 19, 21), dtype=np.uint16)
    # test input volume has to be at least 20x20xNz
    with raises(ValueError, match=r"Input volume is too small"):
        find_somas(volume, res)
    volume = np.zeros((50, 50, 50), dtype=np.uint16)

    # test res should be list
    with raises(TypeError, match=r"should be <class 'list'>"):
        find_somas(volume, res)
    res = [1, "a", 2]
    with raises(
        TypeError, match=r"elements should be \(<class 'int'>, <class 'float'>\)."
    ):
        find_somas(volume, res)
    res = [100, 100.0]
    with raises(ValueError, match=r"Resolution must be three-dimensional"):
        find_somas(volume, res)
    res = [100, 100.0, 0]
    with raises(ValueError, match=r"Resolution must be non-zero at every position"):
        find_somas(volume, res)


##################
### validation ###
##################


def test_detect_output():
    volume_key = "4807349.0_3827990.0_2922565.75_4907349.0_3927990.0_3022565.75"
    mips = {
        1: [597.518465909091, 608.831796875, 1976.8082932692307],
        2: [1195.036931818182, 1217.66359375, 3953.6165865384614],
        3: [2390.073863636364, 2435.3271875, 7907.233173076923],
    }
    soma = np.load(os.path.join(data_dir, "soma.npy"), allow_pickle=True)
    for mip in [1, 2, 3]:
        res = mips[mip]
        print(res)

        volume_coords = np.array(volume_key.split("_")).astype(float)
        volume_vox_min = np.round(np.divide(volume_coords[:3], res)).astype(int)

        img = np.load(os.path.join(data_dir, f"{mip}_volume.npy"), allow_pickle=True)
        label, rel_pred_centroids, out = find_somas(img, res)
        # check output type
        assert type(label) == bool
        assert type(rel_pred_centroids) == np.ndarray
        assert type(out) == np.ndarray
        # check output dimension
        assert rel_pred_centroids.shape[1] == 3
        assert out.shape == (160, 160, 50)
        # check detected somas matche with ground truth
        pred_centroids = np.array(
            [np.multiply(volume_vox_min + c, res) for c in rel_pred_centroids]
        )

        soma_norms = np.linalg.norm(soma, axis=1)
        pred_norms = np.linalg.norm(pred_centroids, axis=1)
        match = np.array(
            [
                min([abs(prediction - soma) for prediction in pred_norms]) < 5e3
                for soma in soma_norms
            ]
        )
        assert match.all()
