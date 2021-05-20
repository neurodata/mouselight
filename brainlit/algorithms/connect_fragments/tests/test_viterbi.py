# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 23:47:23 2021

@author: frede
"""
import numpy as np
import pytest
from grid_generator import grid_gen, grid_builder


def testInit():
    # alg.x == x will return a boolean matrix. The sum should be equal to
    #   dim1 * dim2 * dim3 if every position is True
    img, lbls, _, somas = grid_gen(10)
    alg = grid_builder("0")
    assert np.sum(alg.image == img) == 10 * 10 * 2
    assert np.sum(alg.labels == lbls) == 10 * 10 * 2


def testLineToLineBadQuery():
    alg = grid_builder("0")
    alg1_3D = grid_builder("1_3")
    with pytest.raises(ValueError):
        alg.line_line_dist(2, 2)
        alg1_3D.line_line_dist(2, 2)


def testLineToBlobBadQuery():
    alg = grid_builder("0")
    alg2_3D = grid_builder("2_3")
    with pytest.raises(ValueError):
        alg.line_blob_dist(2, 2)
        alg2_3D.line_blob_dist(2, 2)


def testLineToLine():
    alg = grid_builder("0")
    alg_half = grid_builder("0.5")
    alg1_3D = grid_builder("1_3")
    alg1_3D_half = grid_builder("1_3.5")

    # Test in 2D, pseudo-3D
    # Compare the output of line_line_dist with manually computed distances
    assert np.sqrt((0 - 3) ** 2 + (2 - 3) ** 2) == alg.line_line_dist(1, 3)[0]
    assert np.sqrt((0 - 3) ** 2 + (2 - 3) ** 2) == alg.line_line_dist(3, 1)[0]
    assert np.sqrt((6 - 7) ** 2 + (5 - 7) ** 2) == alg.line_line_dist(3, 4)[0]
    assert np.sqrt((6 - 7) ** 2 + (5 - 7) ** 2) == alg.line_line_dist(4, 3)[0]

    # Test in 2D, pseudo-3D, with resolution scaling of [0.5,0.5,0.5]
    # Compare the output of line_line_dist with manually computed distances
    assert (
        np.sqrt(((0 - 3) / 2) ** 2 + ((2 - 3) / 2) ** 2)
        == alg_half.line_line_dist(1, 3)[0]
    )
    assert (
        np.sqrt(((0 - 3) / 2) ** 2 + ((2 - 3) / 2) ** 2)
        == alg_half.line_line_dist(3, 1)[0]
    )
    assert (
        np.sqrt(((6 - 7) / 2) ** 2 + ((5 - 7) / 2) ** 2)
        == alg_half.line_line_dist(3, 4)[0]
    )
    assert (
        np.sqrt(((6 - 7) / 2) ** 2 + ((5 - 7) / 2) ** 2)
        == alg_half.line_line_dist(4, 3)[0]
    )

    # Test in 3D
    # Compare the output of line_line_dist with manually computed distances
    assert (
        np.sqrt((3 - 96) ** 2 + (0 - 99) ** 2 + (0 - 99) ** 2)
        == alg1_3D.line_line_dist(1, 3)[0]
    )
    assert (
        np.sqrt((96 - 3) ** 2 + (99 - 0) ** 2 + (99 - 0) ** 2)
        == alg1_3D.line_line_dist(3, 1)[0]
    )

    # Test in 3D, with resolution scaling of [0.5,0.5,0.5]
    # Compare the output of line_line_dist with manually computed distances
    assert (
        np.sqrt(((3 - 96) / 2) ** 2 + ((0 - 99) / 2) ** 2 + ((0 - 99) / 2) ** 2)
        == alg1_3D_half.line_line_dist(1, 3)[0]
    )
    assert (
        np.sqrt(((96 - 3) / 2) ** 2 + ((99 - 0) / 2) ** 2 + ((99 - 0) / 2) ** 2)
        == alg1_3D_half.line_line_dist(3, 1)[0]
    )


def testLineToBlob():
    alg = grid_builder("0")
    alg2 = grid_builder("2")
    alg_half = grid_builder("0.5")
    alg2_half = grid_builder("2.5")
    alg2_3D = grid_builder("2_3")
    alg2_3D_half = grid_builder("2_3.5")

    # Test in 2D, pseudo-3D
    # Compare the output of line-blob dist with manually computed distances
    # To a soma
    assert np.sqrt((8 - 9) ** 2 + (7 - 9) ** 2) == alg.line_blob_dist(4, 5)[0]
    # Should go from an endpoint to the closest position on the big blob
    assert np.sqrt((9 - 3) ** 2 + (9 - 15) ** 2) == alg2.line_blob_dist(2, 1)[0]

    # Test in 2D, pseudo-3D, with resolution scaling of [0.5,0.5,0.5]
    # Compare the output of line-blob dist with manually computed distances
    # To a soma
    assert (
        np.sqrt(((8 - 9) / 2) ** 2 + ((7 - 9) / 2) ** 2)
        == alg_half.line_blob_dist(4, 5)[0]
    )
    # Should go from an endpoint to the closest position on the big blob
    assert (
        np.sqrt(((9 - 3) / 2) ** 2 + ((9 - 15) / 2) ** 2)
        == alg2_half.line_blob_dist(2, 1)[0]
    )

    # Test in 3D
    # Should go from an endpoint to the closest position on the blob
    assert (
        np.sqrt((4 - 21) ** 2 + (0 - 21) ** 2 + (0 - 21) ** 2)
        == alg2_3D.line_blob_dist(1, 3)[0]
    )

    # Test in 3D, with resolution scaling of [0.5,0.5,0.5]
    # Should go from an endpoint to the closest position on the blob
    assert (
        np.sqrt(((4 - 21) / 2) ** 2 + ((0 - 21) / 2) ** 2 + ((0 - 21) / 2) ** 2)
        == alg2_3D_half.line_blob_dist(1, 3)[0]
    )


def testPathIntensityCost():
    alg1 = grid_builder("1")
    alg2 = grid_builder("2")

    # Currently a stub, need to double-check what the proper form of the
    # intensity score should be
    assert 1 == alg2.line_int((9, 9, 0), (12, 12, 0), 2, 3)
    assert 0.0061162079510703364 == alg1.line_int((0, 2, 0), (3, 3, 0), 1, 3)


def testConnections():
    alg = grid_builder("0")
    # Test if we are getting the correct connections
    alg.compute_all_dists()
    c = alg.connection_mat

    # 1 to x, x to 1
    np.testing.assert_equal(c[0][1][2], [0, 2, 0])
    np.testing.assert_equal(c[1][1][2], [0, 9, 0])
    np.testing.assert_equal(c[0][2][1], [0, 9, 0])
    np.testing.assert_equal(c[1][2][1], [0, 2, 0])

    np.testing.assert_equal(c[0][1][3], [0, 2, 0])
    np.testing.assert_equal(c[1][1][3], [3, 3, 0])
    np.testing.assert_equal(c[0][3][1], [3, 3, 0])
    np.testing.assert_equal(c[1][3][1], [0, 2, 0])

    np.testing.assert_equal(c[0][1][4], [0, 2, 0])
    np.testing.assert_equal(c[1][1][4], [7, 7, 0])
    np.testing.assert_equal(c[0][4][1], [7, 7, 0])
    np.testing.assert_equal(c[1][4][1], [0, 2, 0])

    # 2 to x, x to 2
    np.testing.assert_equal(c[0][2][3], [0, 9, 0])
    np.testing.assert_equal(c[1][2][3], [3, 3, 0])
    np.testing.assert_equal(c[0][3][2], [3, 3, 0])
    np.testing.assert_equal(c[1][3][2], [0, 9, 0])

    np.testing.assert_equal(c[0][2][4], [0, 9, 0])
    np.testing.assert_equal(c[1][2][4], [7, 8, 0])
    np.testing.assert_equal(c[0][4][2], [7, 8, 0])
    np.testing.assert_equal(c[1][4][2], [0, 9, 0])

    # 3 to x, x to 3
    np.testing.assert_equal(c[0][3][4], [6, 5, 0])
    np.testing.assert_equal(c[1][3][4], [7, 7, 0])
    np.testing.assert_equal(c[0][4][3], [7, 7, 0])
    np.testing.assert_equal(c[1][4][3], [6, 5, 0])

    # 4 to x, x to 4 is covered by other test blocks

    # SOMA CONNECTIONS
    # 5 to x, x to 5
    np.testing.assert_equal(c[0][1][5], [0, 2, 0])
    np.testing.assert_equal(c[1][1][5], [9, 9, 0])
    np.testing.assert_equal(c[0][5][1], [0, 0, 0])
    np.testing.assert_equal(c[1][5][1], [0, 0, 0])

    np.testing.assert_equal(c[0][3][5], [6, 5, 0])
    np.testing.assert_equal(c[1][3][5], [9, 9, 0])
    np.testing.assert_equal(c[0][5][3], [0, 0, 0])
    np.testing.assert_equal(c[1][5][3], [0, 0, 0])

    np.testing.assert_equal(c[0][4][5], [7, 8, 0])
    np.testing.assert_equal(c[1][4][5], [9, 9, 0])
    np.testing.assert_equal(c[0][5][4], [0, 0, 0])
    np.testing.assert_equal(c[1][5][4], [0, 0, 0])

    # Cannot go from blob 2 to soma 5
    np.testing.assert_equal(c[0][2][5], [0, 0, 0])
    np.testing.assert_equal(c[1][2][5], [0, 0, 0])
    np.testing.assert_equal(c[0][5][2], [0, 0, 0])
    np.testing.assert_equal(c[1][5][2], [0, 0, 0])

    # Cost matrix checks
    for i in range(5):
        for j in range(5):
            if i in alg.somas.keys():
                assert np.inf == alg.cost_mat_dist[i, j]
                assert np.inf == alg.cost_mat_int[i, j]
            elif i == j:
                assert np.inf == alg.cost_mat_dist[i, j]
                assert np.inf == alg.cost_mat_int[i, j]


def testEndpoints():
    alg3 = grid_builder("3")
    alg3.frags_to_lines_le_skel([2])
    # Labels 1 and 3 are lines for this 100x100 example

    endpoints1 = np.array(alg3.end_points[1])
    endpoints3 = np.array(alg3.end_points[3])

    # Check if both endpoints are found
    # Note that the manually typed endpoints were identified visually
    # in this simple example.
    assert 1 == endpoints1[1][0]
    assert 1 == endpoints1[1][1]
    assert 0 == endpoints1[1][2]

    assert 48 == endpoints1[0][0]
    assert 48 == endpoints1[0][1]
    assert 0 == endpoints1[0][2]

    assert 51 == endpoints3[1][0]
    assert 51 == endpoints3[1][1]
    assert 0 == endpoints3[1][2]

    assert 97 == endpoints3[0][0]
    assert 97 == endpoints3[0][1]
    assert 0 == endpoints3[0][2]


def testViterbi():
    alg = grid_builder("0")
    alg1 = grid_builder("1")

    def get_cost_arr(paths):
        costs = []
        for p in paths:
            costs.append(p[1][0])
        return costs

    # ========== Viterbi 2D ==========

    alg.compute_all_dists()
    top_path, sorted_paths = alg.viterbi_frag(1, K=4, somas=alg.somas)
    top_path_lbls = top_path[1]
    assert top_path_lbls[0] == 1
    assert top_path_lbls[-1] == 5
    top_path_cost = top_path[0]
    all_path_costs = get_cost_arr(sorted_paths)
    assert top_path_cost == min(all_path_costs)

    # _______ WITH DIFFERENT INTENSITY DATA _______

    alg1.compute_all_dists()
    top_path, sorted_paths = alg1.viterbi_frag(1, K=4, somas=alg1.somas)
    top_path_lbls = top_path[1]
    assert top_path_lbls[0] == 1
    assert top_path_lbls[-1] == 5
    top_path_cost = top_path[0]
    all_path_costs = get_cost_arr(sorted_paths)
    assert top_path_cost == min(all_path_costs)

    # __________ K=20 TEST __________
    top_path, sorted_paths = alg.viterbi_frag(1, K=20, somas=alg.somas)
    top_path_lbls = top_path[1]
    assert top_path_lbls[0] == 1
    assert top_path_lbls[-1] == 5
    top_path_cost = top_path[0]
    all_path_costs = get_cost_arr(sorted_paths)
    assert top_path_cost == min(all_path_costs)


def testViterbi3D():
    alg4_3D = grid_builder("4_3")

    def get_cost_arr(paths):
        costs = []
        for p in paths:
            costs.append(p[1][0])
        return costs

    # ========== Viterbi 3D ==========

    alg4_3D.compute_all_dists()

    # __________ K=13 TEST __________
    top_path, sorted_paths = alg4_3D.viterbi_frag(1, K=13, somas=alg4_3D.somas)
    top_path_lbls = top_path[1]
    assert top_path_lbls[0] == 1
    assert top_path_lbls[-1] == 5
    top_path_cost = top_path[0]
    all_path_costs = get_cost_arr(sorted_paths)
    assert top_path_cost == min(all_path_costs)

    # __________ K=30 TEST __________
    top_path, sorted_paths = alg4_3D.viterbi_frag(1, K=30, somas=alg4_3D.somas)
    top_path_lbls = top_path[1]
    assert top_path_lbls[0] == 1
    assert top_path_lbls[-1] == 5
    top_path_cost = top_path[0]
    all_path_costs = get_cost_arr(sorted_paths)
    assert top_path_cost == min(all_path_costs)
