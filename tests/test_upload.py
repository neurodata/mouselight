import pytest
from brainlit.utils import upload, session
from brainlit.algorithms.generate_fragments import tube_seg
from pathlib import Path
import tifffile as tf
from cloudvolume.lib import Bbox

NUM_RES = 1


@pytest.fixture
def volume_info(num_res=NUM_RES, channel=0):
    """Pytest fixture that gets parameters that many upload.py methods use."""
    top_level = Path(__file__).parents[1] / "data"
    (
        ordered_files,
        bin_paths,
        vox_size,
        tiff_dims,
        origin,
    ) = upload.get_volume_info(str(top_level / "data_octree"), num_res, channel)
    return (
        ordered_files,
        bin_paths,
        vox_size,
        tiff_dims,
        origin,
        top_level,
    )


@pytest.fixture
def paths():
    """Gets common paths for tests running uploads"""
    top_level = Path(__file__).parents[1] / "data"
    input = top_level / "data_octree"
    return top_level, input


####################
### input checks ###
####################


def test_get_volume_info_bad_inputs():
    """Tests that errors are raised when bad inputs are given to upload.get_volume_info."""
    p = str(Path(__file__).parents[1] / "data")
    n = 1
    c = 0
    e = "tif"
    with pytest.raises(TypeError):
        upload.get_volume_info(0, n, c, e)
    with pytest.raises(FileNotFoundError):
        upload.get_volume_info("asdf", n, c, e)
    with pytest.raises(TypeError):
        upload.get_volume_info(p, 0.0, c, e)
    with pytest.raises(ValueError):
        upload.get_volume_info(p, 0, c, e)
    with pytest.raises(TypeError):
        upload.get_volume_info(p, n, 0.0, e)
    with pytest.raises(ValueError):
        upload.get_volume_info(p, n, -1, e)
    with pytest.raises(TypeError):
        upload.get_volume_info(p, n, c, 1)
    with pytest.raises(ValueError):
        upload.get_volume_info(p, n, c, "fff")


def test_create_cloud_volume_bad_inputs(volume_info):
    """Tests that errors are raised when bad inputs are given to upload.create_cloud_volume."""
    (
        _,
        _,
        v,
        i,
        _,
        top_level,
    ) = volume_info
    p = "file://" + str(top_level)
    n = 1
    c = i
    par = False
    l = "image"
    d = "uint16"
    with pytest.raises(TypeError):
        upload.create_cloud_volume(0, i, v, n, c, par, l, d)
    with pytest.raises(NotImplementedError):
        upload.create_cloud_volume("asdf", i, v, n, c, par, l, d)
    with pytest.raises(TypeError):
        upload.create_cloud_volume(p, 0, v, n, c, par, l, d)
    with pytest.raises(ValueError):
        upload.create_cloud_volume(p, [0], v, n, c, par, l, d)
    with pytest.raises(TypeError):
        upload.create_cloud_volume(p, ["a", "b", "c"], v, n, c, par, l, d)
    with pytest.raises(TypeError):
        upload.create_cloud_volume(p, i, 0, n, c, par, l, d)
    with pytest.raises(ValueError):
        upload.create_cloud_volume(p, i, [0], n, c, par, l, d)
    with pytest.raises(TypeError):
        upload.create_cloud_volume(p, i, ["a", "b", "c"], n, c, par, l, d)
    with pytest.raises(TypeError):
        upload.create_cloud_volume(p, i, v, 0.0, c, par, l, d)
    with pytest.raises(ValueError):
        upload.create_cloud_volume(p, i, v, 0, c, par, l, d)
    with pytest.raises(TypeError):
        upload.create_cloud_volume(p, i, v, n, 0, par, l, d)
    with pytest.raises(ValueError):
        upload.create_cloud_volume(p, i, v, n, [0], par, l, d)
    with pytest.raises(TypeError):
        upload.create_cloud_volume(p, i, v, n, ["a", "b", "c"], par, l, d)
    with pytest.raises(TypeError):
        upload.create_cloud_volume(p, i, v, n, c, 0, l, d)
    with pytest.raises(TypeError):
        upload.create_cloud_volume(p, i, v, n, c, par, 0, d)
    with pytest.raises(ValueError):
        upload.create_cloud_volume(p, i, v, n, c, par, "seg", d)
    with pytest.raises(TypeError):
        upload.create_cloud_volume(p, i, v, n, c, par, l, 0)
    with pytest.raises(ValueError):
        upload.create_cloud_volume(p, i, v, n, c, par, l, "uint8")
    with pytest.raises(TypeError):
        upload.create_cloud_volume(p, i, v, n, c, par, l, d, 0)


def test_get_data_ranges_bad_inputs(volume_info):
    """Tests that errors are raised when bad inputs are given to upload.get_data_ranges."""
    _, bin_paths, _, tiff_dims, _, _ = volume_info
    with pytest.raises(TypeError):
        upload.get_data_ranges(0, tiff_dims)
    with pytest.raises(TypeError):
        upload.get_data_ranges(0, tiff_dims)
    with pytest.raises(TypeError):
        upload.get_data_ranges(bin_paths[0], 0)


def test_process_bad_inputs(volume_info):
    """Tests that errors are raised when bad inputs are given to upload.process."""
    fpaths, bin_paths, v, i, o, top = volume_info
    dest = "file://" + str(top / "test_upload")
    vols = upload.create_cloud_volume(dest, i, v, NUM_RES)
    with pytest.raises(TypeError):
        upload.process(0, bin_paths[0][0], vols[0])
    with pytest.raises(FileNotFoundError):
        upload.process("asdf", bin_paths[0][0], vols[0])
    with pytest.raises(TypeError):
        upload.process(fpaths[0][0], 0, vols[0])
    with pytest.raises(ValueError):
        upload.process(fpaths[0][0], ["asdf"], vols[0])
    with pytest.raises(TypeError):
        upload.process(fpaths[0][0], bin_paths[0][0], 0)


def test_upload_volumes_bad_inputs(volume_info):
    """Tests that errors are raised when bad inputs are given to upload.upload_volumes."""
    fpaths, bin_paths, v, i, o, top = volume_info
    n = NUM_RES
    p = False
    c = -1
    root = str(top / "data_octree")
    dest = "file://" + str(top / "test_upload")
    with pytest.raises(TypeError):
        upload.upload_volumes(0, dest, n, p, c)
    with pytest.raises(TypeError):
        upload.upload_volumes(root, 0, n, p, c)
    with pytest.raises(NotImplementedError):
        upload.upload_volumes(root, "asdf", n, p, c)
    with pytest.raises(TypeError):
        upload.upload_volumes(root, dest, 0.0, p, c)
    with pytest.raises(ValueError):
        upload.upload_volumes(root, dest, 0, p, c)
    with pytest.raises(TypeError):
        upload.upload_volumes(root, dest, n, 0, c)
    with pytest.raises(TypeError):
        upload.upload_volumes(root, dest, n, p, 0.0)
    with pytest.raises(ValueError):
        upload.upload_volumes(root, dest, n, p, NUM_RES)


def test_create_skel_segids_bad_inputs(volume_info):
    """Tests that errors are raised when bad inputs are given to upload.create_skel_segids."""
    fpaths, bin_paths, v, i, o, top = volume_info
    swcpath = str(top / "data_octree" / "consensus_swcs")
    with pytest.raises(TypeError):
        upload.create_skel_segids(0, o)
    with pytest.raises(FileNotFoundError):
        upload.create_skel_segids("", o)
    with pytest.raises(TypeError):
        upload.create_skel_segids(swcpath, 0)
    with pytest.raises(ValueError):
        upload.create_skel_segids(swcpath, (0, 0))


def test_upload_segments_bad_inputs(volume_info):
    """Tests that errors are raised when bad inputs are given to upload.upload_segments."""
    fpaths, bin_paths, v, i, o, top = volume_info
    n = NUM_RES
    root = str(top / "data_octree")
    dest = "file://" + str(top / "test_upload_segments")
    with pytest.raises(TypeError):
        upload.upload_volumes(0, dest, n)
    with pytest.raises(TypeError):
        upload.upload_volumes(root, 0, n)
    with pytest.raises(NotImplementedError):
        upload.upload_volumes(root, "asdf", n)
    with pytest.raises(TypeError):
        upload.upload_volumes(root, dest, 0.0)


###################
### upload prep ###
###################


def test_get_volume_info(volume_info):
    """Tests that get_volume_info returns correct parameters."""
    ordered_files, bin_paths, vox_size, tiff_dims, origin, top_level = volume_info
    assert len(ordered_files) == NUM_RES and len(bin_paths) == NUM_RES
    assert (
        len(ordered_files[0]) == 1 and len(bin_paths[0]) == 1
    )  # one file for the lowest resolution
    low_res = tf.imread(str(top_level / "data_octree" / "default.0.tif"))
    image_size = low_res.shape[::-1]
    assert tiff_dims == image_size
    assert len(vox_size) == 3
    assert len(origin) == 3


def test_create_image_layer(volume_info):
    """Tests that create_image_layer returns valid CloudVolumePrecomputed object for image data."""
    _, b, vox_size, tiff_dims, _, top_level = volume_info
    dir = top_level / "test_upload"
    vols = upload.create_cloud_volume(
        dir.as_uri(),
        tiff_dims,
        vox_size,
        num_resolutions=NUM_RES,
        layer_type="image",
    )

    assert len(vols) == NUM_RES  # one vol for each resolution
    for i, vol in enumerate(vols):
        assert vol.scales[-1 - i]["size"] == [(2 ** i) * j for j in tiff_dims]
    assert (dir / "info").is_file()  # contains info file


def test_create_segmentation_layer(volume_info):
    """Tests that create_cloud_volume returns valid CloudVolumePrecomputed object for segmentation data."""
    _, b, vox_size, tiff_dims, _, top_level = volume_info
    dir_segments = top_level / "test_upload_segments"
    vols = upload.create_cloud_volume(
        dir_segments.as_uri(),
        tiff_dims,
        vox_size,
        num_resolutions=NUM_RES,
        layer_type="segmentation",
    )

    assert len(vols) == 1  # always 1 for segementation
    for i, vol in enumerate(vols):
        assert vol.scales[-1 - i]["size"] == [(2 ** i) * j for j in tiff_dims]
    assert (dir_segments / "info").is_file()  # contains info file


def test_create_annotation_layer(volume_info):
    """Tests that create_cloud_volume returns valid CloudVolumePrecomputed object for annotation data."""
    _, b, vox_size, tiff_dims, _, top_level = volume_info
    dir_annotation = top_level / "test_upload_annotations"
    vols = upload.create_cloud_volume(
        dir_annotation.as_uri(),
        tiff_dims,
        vox_size,
        num_resolutions=NUM_RES,
        layer_type="annotation",
    )

    assert len(vols) == 1  # always 1 for segementation
    for i, vol in enumerate(vols):
        assert vol.scales[-1 - i]["size"] == [(2 ** i) * j for j in tiff_dims]
    assert (dir_annotation / "info").is_file()  # contains info file


def test_get_data_ranges(volume_info):
    """Tests that get_data_ranges returns valid ranges."""
    _, bin_paths, _, tiff_dims, _, _ = volume_info
    for res_bins in bin_paths:
        for bin in res_bins:
            x_curr, y_curr, z_curr = 0, 0, 0
            tree_level = len(bin)
            ranges = upload.get_data_ranges(bin, tiff_dims)
            if tree_level == 0:
                assert ranges == ([0, 528], [0, 400], [0, 208])
            else:
                bin = bin[0]
                scale_factor = 1
                x_curr += int(bin[2]) * 528 * scale_factor
                y_curr += int(bin[1]) * 400 * scale_factor
                z_curr += int(bin[0]) * 208 * scale_factor
                assert ranges[0] == [x_curr, x_curr + 528]  # valid x range
                assert ranges[1] == [y_curr, y_curr + 400]  # valid y range
                assert ranges[2] == [z_curr, z_curr + 208]  # valid z range


def test_create_skel_segids(volume_info):
    """Tests that create_skel_segids generates valid skeleton objects."""
    _, _, _, _, origin, top_level = volume_info
    swc_dir = top_level / "data_octree" / "consensus-swcs"
    skels, segids = upload.create_skel_segids(swc_dir.as_posix(), origin)
    assert segids[0] == 2  # we use segment 2 for testing
    assert len(skels[0].vertices) > 0  # should have a positive number of vertices


#################
### uploading ###
#################


def test_upload_segmentation(paths):
    """Ensures that upload_segmentation runs without errors."""
    top_level, input = paths
    dir_segments = top_level / "test_upload_segments"
    upload.upload_segments(input.as_posix(), dir_segments.as_uri(), NUM_RES)
    dir_skel = dir_segments / "skeletons"
    assert (dir_segments / "info").is_file()  # contains info file
    assert (dir_skel / "info").is_file()  # contains skeleton info file
    assert len(sorted(dir_skel.glob("*.gz"))) > 0  # contains uploaded data


def test_upload_volumes_serial(paths):
    """Ensures that upload_volumes runs without errors when `parallel` is False."""
    top_level, input = paths
    dir = top_level / "test_upload" / "serial"
    upload.upload_volumes(input.as_posix(), dir.as_uri(), NUM_RES)
    assert (dir / "info").is_file()  # contains info file
    assert len(sorted(dir.glob("*_*"))) > 0  # contains uploaded data


# def test_upload_annotation(paths):
#     """Ensures that uploading annotations runs without errors.
#     """
#     top_level, input = paths
#     dir = top_level / "test_upload" / "serial"
#     dir_segments = top_level / "test_upload_segments"
#     dir_annotation = top_level / "test_upload_annotations"
#     sess = session.NeuroglancerSession(
#         url=dir.as_uri(),
#         url_segments=dir_segments.as_uri(),
#         url_annotations=dir_annotation.as_uri(),
#         mip=NUM_RES - 1,
#     )
#     img, bounds, vox_list = sess.pull_vertex_list(2, [0, 1], expand=True)
#     labels = tube_seg.tubes_seg(img, vox_list, radius=0.5)
#     sess.push(labels, bounds)
# assert (dir_annotation / "info").is_file()  # contains info file


def test_upload_volumes_parallel(paths):
    """Ensures that upload_volumes runs without errors when `parallel` is True.
    Currently buggy for NUM_RES > 1.
    """
    top_level, input = paths
    dir = Path(top_level) / "test_upload" / "parallel"
    upload.upload_volumes(input.as_posix(), dir.as_uri(), 1, parallel=True)
    assert (dir / "info").is_file()  # contains info file
    assert len(sorted(dir.glob("*_*"))) > 0  # contains uploaded data


##################
### validation ###
##################


def test_serial_download(paths):
    """Tests that downloaded (uploaded serially) data matches original data."""
    top_level, input = paths
    dir = top_level / "test_upload" / "serial"
    dir_segments = top_level / "test_upload_segments"
    ngl_sess = session.NeuroglancerSession(
        url=dir.as_uri(), url_segments=dir_segments.as_uri(), mip=NUM_RES - 1
    )
    orig_img = tf.imread((top_level / "data_octree" / "default.0.tif").as_posix())
    downloaded_img = ngl_sess.pull_bounds_img(Bbox((0, 0, 0), orig_img.shape[::-1]))
    assert downloaded_img.shape == orig_img.shape[::-1]  # same shape
    assert (downloaded_img > 0).any()  # nonzero download
    assert (downloaded_img == orig_img.T).all()  # same img


def test_parallel_download(paths):
    """Tests that downloaded (uploaded in parallel) data matches original data.
    Currently not functional. Deprecated as serial is sufficient.
    """
    # top_level, input = paths
    # dir = top_level / "test_upload" / "parallel"
    # dir_segments = top_level / "test_upload_segments"
    # ngl_sess = session.NeuroglancerSession(
    #     url=dir.as_uri(), url_segments=dir_segments.as_uri(), mip=0
    # )
    # orig_img = tf.imread((top_level / "data_octree" / "default.0.tif").as_posix())
    # downloaded_img = ngl_sess.pull_bounds_img(Bbox((0, 0, 0), orig_img.shape[::-1]))
    # assert downloaded_img.shape == orig_img.shape[::-1]  # same shape
    # assert (downloaded_img > 0).any()  # nonzero download
    # assert (downloaded_img == orig_img.T).all()  # same image
