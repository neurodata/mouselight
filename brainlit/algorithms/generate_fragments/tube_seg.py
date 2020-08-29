import numpy as np
from skimage import draw
import itertools
from scipy.ndimage.morphology import distance_transform_edt
from typing import Optional, List, Tuple, Union
from skimage import draw
from brainlit.utils.util import check_type, check_size, check_iterable_type
from tqdm import tqdm


def pairwise(iterable):
    # Adapted from https://stackoverflow.com/a/5434936
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def draw_sphere(shape, center, radius):
    """
    Generate a sphere of a radius at a point.
    
    Adapted from https://stackoverflow.com/a/56060957
    
    Parameters
    -------
    shape : tuple
        The shape of output array.
    
    center : tuple
        The coordinates for the center of the sphere.
    
    radius : float
        The radius of the sphere.
    
    Returns
    -------
    sphere : numpy.ndarray
        An binary-valued array including a sphere.
    
    """
    coords = np.ogrid[: shape[0], : shape[1], : shape[2]]
    distance = np.sqrt(
        (coords[0] - center[0]) ** 2
        + (coords[1] - center[1]) ** 2
        + (coords[2] - center[2]) ** 2
    )
    sphere = 1 * (distance <= radius)
    return sphere


def draw_tube_from_spheres(img, vertex0, vertex1, radius):
    """
    Generate a segmentation mask of a tube (series of spheres) connecting known vertices.
    
    Parameters
    -------
    img : cloudvolume.volumecutout.VolumeCutout
        The volume to segment.
    
    vertex0 : tuple
        A vertex containing a coordinate within a known segment.
    
    vertex1 : tuple
        A vertex containing a coordinate within a known segment.
    
    radius : float
        The radius of the cylinder.
    
    Returns
    -------
    labels : numpy.ndarray
        An array consisting of the pixelwise segmentation.
    
    """
    line = draw.line_nd(vertex0, vertex1, endpoint=True)
    line = np.array(line).T
    seg = np.zeros(img.shape)
    for pt in line:
        s = draw_sphere(img.shape, pt, radius)
        seg += s

    labels = np.where(seg >= 1, 1, 0)
    return labels


def draw_tube_from_edt(img, vertex0, vertex1, radius):
    """
    Generate a segmentation mask of a tube connecting known vertices.
    
    Parameters
    -------
    img : cloudvolume.volumecutout.VolumeCutout
        The volume to segment.
    
    vertex0 : tuple
        A vertex containing a coordinate within a known segment.
    
    vertex1 : tuple
        A vertex containing a coordinate within a known segment.
    
    radius : float
        The radius of the cylinder.
    
    Returns
    -------
    labels : numpy.ndarray
        An array consisting of the pixelwise segmentation.
    
    """
    line = draw.line_nd(vertex0, vertex1, endpoint=True)
    line_array = np.ones(img.shape, dtype=int)
    line_array[line] = 0
    seg = distance_transform_edt(line_array)
    labels = np.where(seg <= radius, 1, 0)
    return labels


def tubes_seg(img, vertices, radius, spheres=False):
    """
    Generate a segmentation mask of cylinders connecting known vertices.
    
    Parameters
    -------
    img : cloudvolume.volumecutout.VolumeCutout
        The volume to segment.
    
    vertices : list
        The vertices (tuples) each containing a coordinate within a known segment.
    
    radius : float
        The radius of the cylinder.
    spheres : bool
        True if sphere-based segmentation should be used; False for EDT-based segmentation.
    
    Returns
    -------
    labels : numpy.ndarray
        An array consisting of the pixelwise segmentation.
    
    """
    output = np.zeros(img.shape)
    for a, b in pairwise(vertices):
        if spheres:
            output += draw_tube_from_spheres(img, a, b, radius)
        else:
            output += draw_tube_from_edt(img, a, b, radius)
    labels = np.where(output >= 1, 1, 0)
    return labels


def tubes_from_paths(
    size: Tuple[int, int, int],
    paths: List[List[int]],
    radius: Optional[Union[float, int]] = None,
):
    """Constructs tubes from list of paths.
    Returns densely labeled paths within the shape of the image.

    Arguments:
        size: The size of image to consider.
        paths: The list of paths. Each path is a list of points along the path (non-dense).
        radius: The radius of the line to draw. Default is None = 1 pixel wide line.
    """
    check_size(size)
    for path in paths:
        [check_iterable_type(vert, (int, np.integer)) for vert in path]
    if radius is not None:
        check_type(radius, (int, float))
        if radius <= 0:
            raise ValueError(f"Radius {radius} must be positive.")

    def _within_img(line, size):
        arrline = np.array(line).astype(int)
        arrline = arrline[:, arrline[0, :] < size[0]]
        arrline = arrline[:, arrline[0, :] >= 0]
        arrline = arrline[:, arrline[1, :] < size[1]]
        arrline = arrline[:, arrline[1, :] >= 0]
        arrline = arrline[:, arrline[2, :] < size[2]]
        arrline = arrline[:, arrline[2, :] >= 0]
        return (arrline[0, :], arrline[1, :], arrline[2, :])

    coords = [[], [], []]
    for path in tqdm(paths):
        for i in range(len(path) - 1):
            line = draw.line_nd(path[i], path[i + 1])
            line = _within_img(line, size)
            if len(line) > 0:
                coords[0] = np.concatenate((coords[0], line[0]))
                coords[1] = np.concatenate((coords[1], line[1]))
                coords[2] = np.concatenate((coords[2], line[2]))

    try:
        coords = (coords[0].astype(int), coords[1].astype(int), coords[2].astype(int))
    except AttributeError:  # if a list was passed
        coords = (coords[0], coords[1], coords[2])

    if radius is not None:
        line_array = np.ones(size, dtype=int)
        line_array[coords] = 0
        seg = distance_transform_edt(line_array)
        labels = np.where(seg <= radius, 1, 0)
    else:
        labels = np.zeros(size, dtype=int)
        labels[coords] = 1

    return labels
