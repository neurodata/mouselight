import pytest
import brainlit
from brainlit.algorithms.generate_fragments import tube_seg
import numpy as np
from brainlit.utils.session import NeuroglancerSession
from brainlit.utils.swc import graph_to_paths
from skimage import draw
from skimage.morphology import distance_transform_edt
from pathlib import Path

URL = "s3://mouse-light-viz/precomputed_volumes/brain1"
top_level = Path(__file__).parents[1] / "data"
input = (top_level / "data_octree").as_posix()
url = (top_level / "test_upload").as_uri()
URL_SEG = url + "_segments"
URL = url + "/serial"


def test_pairwise():

    """
    For a given iterable array [A1,A2,...,An], test if the function can return a zipped list [(A1,A2),(A2,A3),...,(An-1,An)]
    
    The list should contain n-1 pairs(2-element tuple)
    
    The first element of all the tuples are [A1,A2,...,An-1]
    
    The second element of all the tubles are [A2,A3,...,An]
    """
    n = np.random.randint(4, 9)
    iterable = np.random.randint(10, size=(n, 3))
    pair = list(tube_seg.pairwise(iterable))

    """
    Verify:
   
    I. the number of pairs
    II. the first element of each pair
    III. the second elemnt of each pair
    """
    assert len(pair) == n - 1
    assert (iterable[:-1, :] == [a[0] for a in pair]).all()
    assert (iterable[1:, :] == [b[1] for b in pair]).all()


def test_draw_sphere():

    """
    Test if the function maps all the points located within the given radius of the given center to 1, otherwise 0
    
    The output array should have the same size of input image and binary values
    
    Distance between a point and the given center:
             <= radius (if the point has value 1)
             >  radius (if the point has value 0)
    """
    ngl_session = NeuroglancerSession(url=URL, url_segments=URL_SEG)
    img, _, _ = ngl_session.pull_vertex_list(2, [4], expand=True)
    shape = img.shape
    center = [
        np.random.randint(shape[0]),
        np.random.randint(shape[1]),
        np.random.randint(shape[2]),
    ]
    radius = np.random.randint(1, 4)
    sphere = tube_seg.draw_sphere(shape, center, radius)
    coords = np.where(sphere < 1)
    d_bg = min(np.sum((np.array(coords).T - center) ** 2, axis=1))
    coords = np.where(sphere > 0)
    d_s = max(np.sum((np.array(coords).T - center) ** 2, axis=1))

    """
    Verify:
    
    I. the size of output array
    II. if the output is binary-valued
    III. minimum distance between 0-valued points and the center is greater than radius
    IV. maximum distance between 1-valued points and the center is less than or equal to radius
    """
    assert sphere.shape == shape
    assert np.unique(sphere).all() in [0, 1]
    assert d_bg > radius ** 2
    assert d_s <= radius ** 2


def test_draw_tube_spheres():

    """
    Test if the function maps all the points within the radius of a segment line (defined by 2 given points) to 1, otherwise 0
    
    The output array should have the same size of input image and binary values
    
    Distance between a point and the segment line:
             <= radius (if the point has value 1)
             >  radius (if the point has value 0)        
    """
    ngl_session = NeuroglancerSession(url=URL, url_segments=URL_SEG)
    img, _, _ = ngl_session.pull_vertex_list(2, [4], expand=True)
    shape = img.shape
    vertex0 = [
        np.random.randint(shape[0] / 2),
        np.random.randint(shape[1]),
        np.random.randint(shape[2]),
    ]
    vertex1 = [
        np.random.randint(shape[0] / 2, shape[0]),
        np.random.randint(shape[1]),
        np.random.randint(shape[2]),
    ]
    radius = np.random.randint(1, 4)
    labels = tube_seg.draw_tube_from_spheres(img, vertex0, vertex1, radius)
    line = draw.line_nd(vertex0, vertex1, endpoint=True)
    coords = np.where(labels < 1)
    d_bg = max(shape)
    for pt in np.array(coords).T:
        distance_min = min(np.sum((np.array(line).T - pt) ** 2, axis=1))
        d_bg = min(distance_min, d_bg)
    coords = np.where(labels > 0)
    d_tube = 0
    for pt in np.array(coords).T:
        distance_min = min(np.sum((np.array(line).T - pt) ** 2, axis=1))
        d_tube = max(distance_min, d_tube)

    """
    Verify:
    
    I. the size of output array
    II. if the output is binary-valued
    III. minimum distance between 0-valued points and the segment line is greater than radius
    IV. maximum distance between 1-valued points and the segment line is less than or equal to radius
    """
    assert labels.shape == shape
    assert np.unique(labels).all() in [0, 1]
    assert d_bg > radius ** 2
    assert d_tube <= radius ** 2


def test_draw_tube_edt():

    """
    Test if the function maps all the points within the radius of a segment line (defined by 2 given points) to 1, otherwise 0
    
    The output array should have the same size of input image and binary values
    
    Distance between a point and the segment line:
             <= radius (if the point has value 1)
             >  radius (if the point has value 0)        
    """
    ngl_session = NeuroglancerSession(url=URL, url_segments=URL_SEG)
    img, _, _ = ngl_session.pull_vertex_list(2, [4], expand=True)
    shape = img.shape
    vertex0 = [
        np.random.randint(shape[0] / 2),
        np.random.randint(shape[1]),
        np.random.randint(shape[2]),
    ]
    vertex1 = [
        np.random.randint(shape[0] / 2, shape[0]),
        np.random.randint(shape[1]),
        np.random.randint(shape[2]),
    ]
    radius = np.random.randint(1, 4)
    labels = tube_seg.draw_tube_from_edt(img, vertex0, vertex1, radius)
    line = draw.line_nd(vertex0, vertex1, endpoint=True)
    coords = np.where(labels < 1)
    d_bg = max(shape)
    for pt in np.array(coords).T:
        distance_min = min(np.sum((np.array(line).T - pt) ** 2, axis=1))
        d_bg = min(distance_min, d_bg)
    coords = np.where(labels > 0)
    d_tube = 0
    for pt in np.array(coords).T:
        distance_min = min(np.sum((np.array(line).T - pt) ** 2, axis=1))
        d_tube = max(distance_min, d_tube)

    """
    Verify:
    
    I. the size of output array
    II. if the output is binary-valued
    III. minimum distance between 0-valued points and the segment line is greater than radius
    IV. maximum distance between 1-valued points and the segment line is less than or equal to radius
    """
    assert labels.shape == shape
    assert np.unique(labels).all() in [0, 1]
    assert d_bg > radius ** 2
    assert d_tube <= radius ** 2


def test_tubes_seg():

    """
    Test if the function maps all the points within the radius of polyline (defined by given vertices) to 1, otherwise 0
    
    The output array should have the same size of input image and binary values
    
    Distance between a point and the polyline:
             <= radius (if the point has value 1)
             >  radius (if the point has value 0)
    """
    ngl_session = NeuroglancerSession(url=URL, url_segments=URL_SEG)
    img, _, _ = ngl_session.pull_vertex_list(2, [4], expand=True)
    shape = img.shape
    vertices = np.random.randint(min(shape), size=(4, 3))
    radius = np.random.randint(1, 4)
    labels = tube_seg.tubes_seg(img, vertices, radius)
    point = np.empty((3, 0), dtype=int)
    for i in range(3):
        lines = draw.line_nd(vertices[i], vertices[i + 1], endpoint=True)
        point = np.concatenate((point, np.array(lines)), axis=1)
    coords = np.where(labels < 1)
    d_bg = max(shape)
    for pt in np.array(coords).T:
        distance_min = min(np.sum((point.T - pt) ** 2, axis=1))
        d_bg = min(distance_min, d_bg)
    coords = np.where(labels > 0)
    d_tube = 0
    for pt in np.array(coords).T:
        distance_min = min(np.sum((point.T - pt) ** 2, axis=1))
        d_tube = max(distance_min, d_tube)

    """
    Verify:
    
    I. the size of output array
    II. if the output is binary-valued
    III. minimum distance between 0-valued points and the polyline is greater than radius
    IV. maximum distance between 1-valued points and the polyline is less than or equal to radius
    """
    assert labels.shape == shape
    assert np.unique(labels).all() in [0, 1]
    assert d_bg > radius ** 2
    assert d_tube <= radius ** 2


def test_tubes_from_paths_bad_inputs():
    """Tests that the tubes_from_paths method raises errors when given bad inputs.
    """
    sess = NeuroglancerSession(URL, 0, URL_SEG)
    img, bbox, verts = sess.pull_voxel(2, 300, radius=5)  # A valid bbox with data.
    G = sess.get_segments(2, bbox)
    bbox = bbox.to_list()
    paths = graph_to_paths(G)  # valid paths
    size = np.subtract(bbox[3:], bbox[:3])
    with pytest.raises(TypeError):
        tube_seg.tubes_from_paths("asdf", paths)
    with pytest.raises(ValueError):
        tube_seg.tubes_from_paths((-1, -1, -1), paths)
    with pytest.raises(TypeError):
        tube_seg.tubes_from_paths(size, "asdf")
    with pytest.raises(TypeError):
        tube_seg.tubes_from_paths(size, [[0, 0, "asdf"]])
    with pytest.raises(TypeError):
        tube_seg.tubes_from_paths(size, paths, radius="asdf")
    with pytest.raises(ValueError):
        tube_seg.tubes_from_paths(size, paths, radius=-1)


def test_tubes_from_paths():
    """Tests that, given valid paths, valid tubes are created.
    """
    sess = NeuroglancerSession(URL, 0, URL_SEG)
    img, bbox, verts = sess.pull_voxel(2, 300, radius=5)  # A valid bbox with data.
    G = sess.get_segments(2, bbox)
    bbox = bbox.to_list()
    paths = graph_to_paths(G)  # valid paths
    size = np.subtract(bbox[3:], bbox[:3])
    tubes = tube_seg.tubes_from_paths(size, paths)
    assert (tubes != 0).any()


def test_tubes_exact():
    """Tests that exact pixels are filled in.
    """
    img = np.zeros((10, 10, 10))
    verts = [[5, 5, 0], [5, 5, 10]]
    tubes = tube_seg.tubes_from_paths(img.shape, [verts])
    assert tubes.shape == img.shape
    assert (tubes[5, 5, :] == 1).all()
    for i in range(10):  # set middle column to zero
        tubes[5, 5, i] = 0
    assert (tubes == 0).all()  # now everything should be zero


def test_tubes_radius_exact():
    """Tests that exact pixels are filled with with specific radius.
    """
    img = np.zeros((10, 10, 10))
    verts = [[0, 5, 5], [10, 5, 5]]
    indices = distance_transform_edt(verts[0], verts[1], return_indices=True)
    tubes = tube_seg.tubes_from_paths(img.shape, [verts], radius=2)
    assert tubes.shape == img.shape
    assert (tubes[indices] == 1).all()
    assert (tubes[~indices] == 0).all()
