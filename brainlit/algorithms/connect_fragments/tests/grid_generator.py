# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 21:39:28 2021

@author: frede
"""

import numpy as np
from skimage import measure
from brainlit.algorithms.connect_fragments.dynamic_programming_viterbi import (
    viterbi_algorithm,
)


def grid_gen(grid_id=10):
    if grid_id == 10:
        """Generates a 10x10x2 grid with a small assortment of labels"""
        # 10x10x2 image, with 1 color channel greyscale
        grid = np.zeros((10, 10, 2))
        labels = np.zeros((10, 10, 2))
        grid[0, 9, 0] = 255
        grid[0, 0, 0] = 255
        grid[0, 1, 0] = 255
        grid[0, 2, 0] = 255
        grid[3, 3, 0] = 255
        grid[4, 3, 0] = 255
        grid[5, 3, 0] = 255
        grid[5, 4, 0] = 255
        grid[6, 4, 0] = 255
        grid[6, 5, 0] = 255
        grid[7, 7, 0] = 255
        grid[7, 8, 0] = 255
        grid[9, 9, 0] = 255  # Soma

        # For simplicity, only the 0th layer of grid has labels
        # Label the components
        labels, num = measure.label(grid, return_num=True)

        # Create intensity data
        grid[0, 9, 0] = 255  # Frag

        grid[0, 0, 0] = 150  # Frag
        grid[0, 1, 0] = 150  # Frag
        grid[0, 2, 0] = 150  # Frag

        grid[1, 2, 0] = 150
        grid[1, 3, 0] = 150
        grid[2, 3, 0] = 175

        grid[3, 3, 0] = 200  # Frag
        grid[4, 3, 0] = 200  # Frag
        grid[5, 3, 0] = 200  # Frag
        grid[5, 4, 0] = 200  # Frag
        grid[6, 4, 0] = 200  # Frag
        grid[6, 5, 0] = 200  # Frag

        grid[6, 6, 0] = 215
        grid[7, 6, 0] = 215

        grid[7, 7, 0] = 225  # Frag
        grid[7, 8, 0] = 225  # Frag

        grid[8, 8, 0] = 240
        grid[8, 9, 0] = 250

        grid[9, 9, 0] = 255  # Soma

        # We'll have the bottom-right corner, which is labeled 5, be the soma
        somas = {5: [(9, 9, 0)]}

    if grid_id == 102:
        """Generates a 10x10x2 grid with a small assortment of labels"""
        # 10x10x2 image, with 1 color channel greyscale
        grid = np.zeros((10, 10, 2))
        labels = np.zeros((10, 10, 2))
        grid[0, 9, 0] = 255
        grid[0, 0, 0] = 255
        grid[0, 1, 0] = 255
        grid[0, 2, 0] = 255
        grid[3, 3, 0] = 255
        grid[4, 3, 0] = 255
        grid[5, 3, 0] = 255
        grid[5, 4, 0] = 255
        grid[6, 4, 0] = 255
        grid[6, 5, 0] = 255
        grid[7, 7, 0] = 255
        grid[7, 8, 0] = 255
        grid[9, 9, 0] = 255  # Soma

        # For simplicity, only the 0th layer of grid has labels
        # Label the components
        labels, num = measure.label(grid, return_num=True)

        # Create intensity data, similar to the other grid10 but it has an
        # intensity anomaly
        grid[0, 9, 0] = 255  # Blob

        grid[0, 0, 0] = 150  # Frag
        grid[0, 1, 0] = 150  # Frag
        grid[0, 2, 0] = 150  # Frag

        grid[1, 2, 0] = 150
        grid[1, 3, 0] = 150
        grid[2, 3, 0] = 175

        grid[3, 3, 0] = 255  # Frag
        grid[4, 3, 0] = 255  # Frag
        grid[5, 3, 0] = 255  # Frag
        grid[5, 4, 0] = 255  # Frag
        grid[6, 4, 0] = 255  # Frag
        grid[6, 5, 0] = 255  # Frag

        grid[6, 6, 0] = 40
        grid[7, 6, 0] = 40

        grid[7, 7, 0] = 50  # Frag
        grid[7, 8, 0] = 50  # Frag

        grid[8, 8, 0] = 50
        grid[8, 9, 0] = 50

        grid[9, 9, 0] = 255  # Soma

        # We'll have the bottom-right corner, which is labeled 5, be the soma
        somas = {5: [(9, 9, 0)]}

    if grid_id == 20:
        """Generates a 20x20x2 grid with a small assortment of labels"""
        # 20x20x2 image, with 1 color channel greyscale
        grid = np.zeros((20, 20, 2))
        labels = np.zeros((20, 20, 2))
        # Create some diagonal lines
        for i in range(1, 10):
            grid[i, i, 0] = 200

        # Create some diagonal lines
        for i in range(12, 16):
            grid[i, i, 0] = 175

        # Create a "blob"
        grid[0, 16, 0] = 225
        grid[0, 17, 0] = 225
        grid[0, 18, 0] = 225
        for i in range(1, 4):
            grid[i, 15, 0] = 225
            grid[i, 16, 0] = 225
            grid[i, 17, 0] = 225
            grid[i, 18, 0] = 225
            grid[i, 19, 0] = 225
        grid[4, 16, 0] = 225
        grid[4, 17, 0] = 225
        grid[4, 18, 0] = 225

        # Label a soma
        grid[19, 19, 0] = 255
        # For simplicity, only the 0th layer of grid has labels
        # Label the components
        labels, num = measure.label(grid, return_num=True)

        # We'll have the bottom-right corner, which is labeled 4, be the soma
        somas = {4: [(19, 19, 0)]}

    if grid_id == 100:
        """Generates a 100x100x2 grid with 2 labels along the diagonal"""
        """ and one soma """
        # 100x100x2 image, with 1 color channel greyscale
        grid = np.zeros((100, 100, 2))
        labels = np.zeros((100, 100, 2))
        # Create some diagonal lines
        for i in range(1, 49):
            grid[i, i, 0] = 200

        # Create some diagonal lines
        for i in range(51, 98):
            grid[i, i, 0] = 175
        # Label a soma
        grid[10, 30, 0] = 255
        # For simplicity, only the 0th layer of grid has labels
        # Label the components
        labels, num = measure.label(grid, return_num=True)
        somas = {2: [(10, 30, 0)]}

    return grid, labels, num, somas


def grid_gen3D(grid_id=10):
    if grid_id == 100:
        grid = np.zeros((100, 100, 100))
        labels = np.zeros((100, 100, 100))
        grid[0, 0, 0] = 255
        grid[1, 0, 0] = 255
        grid[2, 0, 0] = 255
        grid[3, 0, 0] = 255

        grid[99, 99, 99] = 255
        grid[98, 99, 99] = 255
        grid[97, 99, 99] = 255
        grid[96, 99, 99] = 255

        # Mark a soma
        grid[0, 0, 99] = 255

        somas = {2: [(0, 0, 99)]}
        labels, num = measure.label(grid, return_num=True)

    if grid_id == 25:
        grid = np.zeros((25, 25, 25))
        labels = np.zeros((25, 25, 25))

        # Create a line
        for i in range(0, 5):
            grid[i, 0, 0] = 255

        # Create a cube for a blob
        for i in range(20, 25):
            for j in range(20, 25):
                for k in range(20, 25):
                    grid[i, i, i] = 255
                    grid[i, i, i] = 255
                    grid[i, i, i] = 255
                    grid[i, i, i] = 255
                    grid[i, i, i] = 255

        # Remove the corners to make it more blob-like
        grid[20, 20, 20] = 0
        grid[20, 24, 20] = 0
        grid[20, 20, 24] = 0
        grid[20, 24, 24] = 0
        grid[24, 20, 24] = 0
        grid[24, 20, 20] = 0
        grid[24, 20, 24] = 0
        grid[24, 24, 24] = 0
        grid[24, 24, 24] = 0

        # Mark a soma
        grid[0, 0, 24] = 255

        somas = {2: [(0, 0, 24)]}

        labels, num = measure.label(grid, return_num=True)

    if grid_id == 101:
        grid = np.zeros((100, 100, 100))
        labels = np.zeros((100, 100, 100))
        for i in range(1, 90):
            grid[i, 0, 0] = 240
        for j in range(5, 90):
            grid[0, j, 0] = 200
        for k in range(10, 70):
            grid[0, 0, k] = 160

        grid[99, 99, 99] = 255
        labels, num = measure.label(grid, return_num=True)
        print("Soma:", labels[99, 99, 99])
        somas = {3: [(99, 99, 99)]}

    if grid_id == 15:
        grid = np.zeros((100, 100, 100))
        labels = np.zeros((100, 100, 100))
        for i in range(0, 4):
            grid[i, 0, 0] = 150

        grid[4, 3, 0] = 125
        grid[4, 3, 1] = 125
        grid[4, 4, 1] = 125
        grid[4, 4, 2] = 125
        grid[4, 5, 2] = 125
        grid[4, 6, 2] = 125
        grid[4, 7, 2] = 125

        grid[6, 8, 4] = 175
        grid[6, 9, 4] = 175
        grid[6, 10, 4] = 175
        grid[7, 10, 4] = 175
        grid[7, 10, 5] = 175
        grid[7, 10, 6] = 175
        grid[7, 10, 7] = 175

        grid[8, 12, 7] = 225
        grid[8, 12, 8] = 225
        grid[8, 12, 9] = 225
        grid[8, 12, 10] = 225
        grid[9, 12, 10] = 225
        grid[10, 12, 10] = 225
        grid[11, 13, 10] = 225
        grid[12, 13, 10] = 225
        grid[12, 13, 11] = 225

        grid[14, 14, 14] = 255
        labels, num = measure.label(grid, return_num=True)
        somas = {5: [(14, 14, 14)]}

        # Adding intensity data
        grid[3, 1, 0] = 130
        grid[3, 2, 0] = 130
        grid[3, 3, 0] = 130

        grid[4, 7, 2] = 150
        grid[4, 8, 2] = 150
        grid[4, 8, 3] = 150
        grid[5, 8, 3] = 150
        grid[6, 8, 3] = 150

        grid[7, 10, 7] = 200
        grid[8, 10, 7] = 200
        grid[8, 11, 7] = 200

        grid[12, 14, 11] = 245
        grid[13, 14, 11] = 245
        grid[13, 14, 12] = 245
        grid[14, 14, 13] = 245

    return grid, labels, num, somas


def grid_builder(grid_id="0"):
    """Setting up the image environment"""
    alg = None
    if (grid_id == "0") or (grid_id == "0.5"):
        img, lbls, _, somas = grid_gen(10)

        # For the timebeing, manually do the endpoints
        # NOTE: no endpoints for 2 or 5 because they are "blobs"
        endpoints = {}
        endpoints[1] = ((0, 0, 0), (0, 2, 0))
        endpoints[3] = ((3, 3, 0), (6, 5, 0))
        endpoints[4] = ((7, 7, 0), (7, 8, 0))

        if grid_id == "0":
            alg = viterbi_algorithm(img, lbls, somas, [1, 1, 1])
            alg.end_points = endpoints

        if grid_id == "0.5":
            alg = viterbi_algorithm(img, lbls, somas, [0.5, 0.5, 0.5])
            alg.end_points = endpoints

    if (grid_id == "2") or (grid_id == "2.5"):
        img, lbls, _, somas = grid_gen(20)
        # manually do the endpoints
        # NOTE: no endpoints for 1 or 4 because they are blobs
        endpoints = {}
        endpoints[2] = ((1, 1, 0), (9, 9, 0))
        endpoints[3] = ((12, 12, 0), (15, 15, 0))

        if grid_id == "2":
            alg = viterbi_algorithm(img, lbls, somas, [1, 1, 1])
            alg.end_points = endpoints

        if grid_id == "2.5":
            alg = viterbi_algorithm(img, lbls, somas, [0.5, 0.5, 0.5])
            alg.end_points = endpoints

    if grid_id == "3":
        # 100x100 example for endpoints testing
        img, lbls, _, somas = grid_gen(100)
        alg = viterbi_algorithm(img, lbls, somas, [1, 1, 1])
        alg.end_points = {}  # Leave empty, will solve with endpoint function

    if (grid_id == "1_3") or (grid_id == "1_3.5"):
        img, lbls, _, somas = grid_gen3D(100)
        endpoints = {}
        endpoints[1] = ((0, 0, 0), (3, 0, 0))
        endpoints[3] = ((99, 99, 99), (96, 99, 99))

        if grid_id == "1_3":
            alg = viterbi_algorithm(img, lbls, somas, [1, 1, 1])
            alg.end_points = endpoints

        if grid_id == "1_3.5":
            alg = viterbi_algorithm(img, lbls, somas, [0.5, 0.5, 0.5])
            alg.end_points = endpoints

    if (grid_id == "2_3") or (grid_id == "2_3.5"):
        (
            img,
            lbls,
            _,
            somas,
        ) = grid_gen3D(25)
        endpoints = {}
        endpoints[1] = ((0, 0, 0), (4, 0, 0))

        if grid_id == "2_3":
            alg = viterbi_algorithm(img, lbls, somas, [1, 1, 1])
            alg.end_points = endpoints

        if grid_id == "2_3.5":
            alg = viterbi_algorithm(img, lbls, somas, [0.5, 0.5, 0.5])
            alg.end_points = endpoints

    if grid_id == "3_3":
        # 100x100 example for endpoints testing
        img, lbls, _, somas = grid_gen3D(101)
        alg = viterbi_algorithm(img, lbls, somas, [1, 1, 1])
        alg.end_points = {}  # Leave empty, will solve with endpoint function

    if grid_id == "1":
        # 10x10 high intensity loop test
        img, lbls, _, somas = grid_gen(102)

        # For the timebeing, manually do the endpoints
        # NOTE: no endpoints for 2 or 5 because they are "blobs"
        endpoints = {}
        endpoints[1] = ((0, 0, 0), (0, 2, 0))
        endpoints[3] = ((3, 3, 0), (6, 5, 0))
        endpoints[4] = ((7, 7, 0), (7, 8, 0))
        alg = viterbi_algorithm(img, lbls, somas, [1, 1, 1])
        alg.end_points = endpoints

    if grid_id == "4_3":
        # 100x100 viterbi test
        img, lbls, _, somas = grid_gen3D(15)

        # For the timebeing, manually do the endpoints
        # NOTE: no endpoints for 2 or 5 because they are "blobs"
        endpoints = {}
        endpoints[1] = ((0, 0, 0), (3, 0, 0))
        endpoints[2] = ((4, 3, 0), (4, 7, 2))
        endpoints[3] = ((6, 8, 4), (7, 10, 7))
        endpoints[4] = ((8, 12, 7), (12, 13, 11))
        alg = viterbi_algorithm(img, lbls, somas, [1, 1, 1])
        alg.end_points = endpoints

    return alg
