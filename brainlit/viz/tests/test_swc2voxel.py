from brainlit.viz.swc2voxel import Bresenham3D
from pytest import raises

####################
### input checks ###
####################


def test_Bresenham3D_bad_input():
    # test all the inputs must be integer
    [x1, y1, z1, x2, y2, z2] = [1, 2, 3, 4, 5, 6]
    with raises(TypeError):
        Bresenham3D(x1 + 0.5, y1, z1, x2, y2, z2)
    with raises(TypeError):
        Bresenham3D(x1, y1 + 0.5, z1, x2, y2, z2)
    with raises(TypeError):
        Bresenham3D(x1, y1, z1 + 0.5, x2, y2, z2)
    with raises(TypeError):
        Bresenham3D(x1, y1, z1, x2 + 0.5, y2, z2)
    with raises(TypeError):
        Bresenham3D(x1, y1, z1, x2, y2 + 0.5, z2)
    with raises(TypeError):
        Bresenham3D(x1, y1, z1, x2, y2, z2 + 0.5)


####################
#### validation ####
####################


def test_Bresenham3D():
    # use examples adapted from https://www.geeksforgeeks.org/bresenhams-algorithm-for-3-d-line-drawing/

    # example 1
    [x1, y1, z1, x2, y2, z2] = [-1, 1, 1, 5, 3, -1]
    xlist = [-1, 0, 1, 2, 3, 4, 5]
    ylist = [1, 1, 2, 2, 2, 3, 3]
    zlist = [1, 1, 0, 0, 0, -1, -1]
    t_x, t_y, t_z = Bresenham3D(x1, y1, z1, x2, y2, z2)
    assert t_x == xlist
    assert t_y == ylist
    assert t_z == zlist

    # example 2
    [x1, y1, z1, x2, y2, z2] = [0, -7, -3, -5, 2, -1]
    xlist = [0, -1, -1, -2, -2, -3, -3, -4, -4, -5]
    ylist = [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2]
    zlist = [-3, -3, -3, -2, -2, -2, -2, -1, -1, -1]
    t_x, t_y, t_z = Bresenham3D(x1, y1, z1, x2, y2, z2)
    assert t_x == xlist
    assert t_y == ylist
    assert t_z == zlist

    # example 3
    [x1, y1, z1, x2, y2, z2] = [0, -3, -7, -5, -1, 2]
    xlist = [0, -1, -1, -2, -2, -3, -3, -4, -4, -5]
    ylist = [-3, -3, -3, -2, -2, -2, -2, -1, -1, -1]
    zlist = [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2]
    t_x, t_y, t_z = Bresenham3D(x1, y1, z1, x2, y2, z2)
    assert t_x == xlist
    assert t_y == ylist
    assert t_z == zlist
