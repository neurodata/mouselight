import numpy as np
from scipy import ndimage as ndi
from skimage import draw
from brainlit.utils.util import check_type


def snap_points(img, points, radius=[3, 3, 3]):
    """Moves neuron marker points to the highest intensity within a certain radius

    Arguments:
        img {3d array} -- image
        points {pandas dataframe} -- dataframe with swc points as output by combine_swc_img.points2voxel

    Keyword Arguments:
        radius {list} -- voxel radius within which to search for highest intensity (default: {[3,3,3]})

    Returns:
        [pandas dataframe] -- dataframe with same format as points, with new xvox, yvox, zvox values (Note: x,y,z, columns are unchanged)
    """
    points_new = points.copy()
    for idx, pt in points.iterrows():
        voxel = pt[["xvox", "yvox", "zvox"]].values.astype(int)
        lower = voxel - radius
        upper = voxel + radius + 1  # inclusive
        subsection = img[lower[0] : upper[0], lower[1] : upper[1], lower[2] : upper[2]]
        voxel_new = np.unravel_index(np.argmax(subsection, axis=None), subsection.shape)
        voxel_new = lower + voxel_new
        points_new.at[idx, "xvox"] = voxel_new[0]
        points_new.at[idx, "yvox"] = voxel_new[1]
        points_new.at[idx, "zvox"] = voxel_new[2]
    return points_new


def point_threshold(img, points):
    """Threshold image according to the minimum intensity of a set of points

    Arguments:
        img {3d array} -- image
        points {pandas dataframe} -- dataframe with swc points as output by combine_swc_img.points2voxel

    Returns:
        [3d array] -- binary mask from thresholding
        [int] -- threshold value
    """
    voxels = points[["xvox", "yvox", "zvox"]].values
    thresh = np.min(img[[voxels[:, 0]], [voxels[:, 1]], [voxels[:, 2]]])
    im_sd = np.std(img)
    thresh = thresh  # - 2*im_sd
    thresholded = np.where(img > thresh, 1, 0)

    return thresholded, thresh


def remove_small_components(img, size=20):
    """Remove components from binary mask that are small

    Arguments:
        img {3d array} -- image

    Keyword Arguments:
        size {int} -- minimum component size (default: {20})

    Returns:
        [3d array] -- binary mask with small components removed
    """
    label_im, nb_labels = ndi.label(img)
    sizes = ndi.sum(img, label_im, range(nb_labels + 1))
    mask = sizes > size
    binary_img = mask[label_im]
    return binary_img


def skeletonize(img, points):
    """Draw lines between points that are connected to produce binary mask

    Arguments:
        img {3d array} -- image
        points {pandas dataframe} -- dataframe with swc points as output by combine_swc_img.points2voxel

    Returns:
        [3d array] -- binary mask showing skeletonization between points
    """
    mask = 0 * img
    print(points)
    for idx, child in points.iterrows():
        parent_idx = child["parent"]
        if parent_idx != -1 and parent_idx in points["sample"].values:
            parent_pt = points[points["sample"] == [parent_idx]]
            xs, ys, zs = Bresenham3D(
                parent_pt.xvox.item(),
                parent_pt.yvox.item(),
                parent_pt.zvox.item(),
                child.xvox,
                child.yvox,
                child.zvox,
            )

            mask[xs, ys, zs] = 1
    return mask


def skeleton_threshold_intersect(img, points):
    """Compute intersection between two masks: thresholded image and skeletonization of points

    Arguments:
        img {3d array} -- image
        points {pandas dataframe} -- dataframe with swc points as output by combine_swc_img.points2voxel

    Returns:
        [3d array] -- binary mask of intersection
        [int] -- when the threshold is lowered to obtain a single connected component, this indicates the number of iterations used
    """
    skel = skeletonize(img, points)
    dilated = ndi.morphology.binary_dilation(skel, iterations=5)
    point_thresholded, thresh = point_threshold(img, points)
    point_thresholded = remove_small_components(point_thresholded)

    intersect = 0 * dilated
    idxs = (dilated == 1) & (point_thresholded == 1)
    intersect[idxs] = 1

    intersect_iter = intersect.copy()
    _, num_c = ndi.measurements.label(intersect)
    std = np.std(img)
    print("Iterating...")
    counter = 0
    """
    while num_c > 1:
        if np.mod(counter,2) == 0:
            dilated = ndi.morphology.binary_dilation(dilated,iterations=1)
        else:
            thresh = thresh - 0.25*std
            point_thresholded = np.where(img > thresh, 1, 0)
            point_thresholded = remove_small_components(point_thresholded)
        idxs = ((dilated==1) & (point_thresholded == 1))
        intersect_iter[idxs] = 1
        _, num_c = ndi.measurements.label(intersect_iter)
        counter += 1 
    """
    return intersect, intersect_iter


def Bresenham3D(x1, y1, z1, x2, y2, z2):
    """Takes two coordinates and gives the set of coordinates that connects them with a straight line

    Adapted from https://www.geeksforgeeks.org/bresenhams-algorithm-for-3-d-line-drawing/

    Arguments:
        x1 {int} -- first x coodinate
        y1 {int} -- first y coodinate
        z1 {int} -- first z coodinate
        x2 {int} -- second x coodinate
        y2 {int} -- second y coodinate
        z2 {int} -- second z coodinate

    Returns:
        [list] -- list of x coordinate connecting the points
        [list] -- list of y coordinate connecting the points
        [list] -- list of z coordinate connecting the points
    """
    check_type(x1, int)
    check_type(y1, int)
    check_type(z1, int)
    check_type(x2, int)
    check_type(y2, int)
    check_type(z2, int)

    xlist = []
    xlist.append(x1)
    ylist = []
    ylist.append(y1)
    zlist = []
    zlist.append(z1)

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    if x2 > x1:
        xs = 1
    else:
        xs = -1
    if y2 > y1:
        ys = 1
    else:
        ys = -1
    if z2 > z1:
        zs = 1
    else:
        zs = -1

    # Driving axis is X-axis"
    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x1 != x2:
            x1 += xs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz

            xlist.append(x1)
            ylist.append(y1)
            zlist.append(z1)

    # Driving axis is Y-axis"
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y1 != y2:
            y1 += ys
            if p1 >= 0:
                x1 += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            xlist.append(x1)
            ylist.append(y1)
            zlist.append(z1)

    # Driving axis is Z-axis"
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z1 != z2:
            z1 += zs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            xlist.append(x1)
            ylist.append(y1)
            zlist.append(z1)
    return xlist, ylist, zlist
