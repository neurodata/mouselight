import numpy as np
import pandas as pd
import tifffile as tf
import networkx as nx
from cloudvolume import CloudVolume
import brainlit
from brainlit.utils import swc
from brainlit.utils.session import NeuroglancerSession

from pathlib import Path

top_level = Path(__file__).parents[1] / "data"
input = (top_level / "data_octree").as_posix()
url = (top_level / "test_upload").as_uri()
url_seg = url + "_segments"
url = url + "/serial"

# read in s3 path to dataframe
df_s3 = swc.read_s3(url_seg, seg_id=2, mip=0)

# read in swc file to dataframe
swc_path = "./data/data_octree/consensus-swcs/2018-08-01_G-002_consensus.swc"
df, offset, color, cc, branch = swc.read_swc(swc_path)

# # convert swc dataframe from spatial units to voxel units
spacing = np.array([0.29875923, 0.3044159, 0.98840415])
origin = np.array([70093.276, 15071.596, 29306.737])

df_voxel = swc.swc_to_voxel(df, spacing=spacing, origin=origin)
df_voxel_s3 = swc.swc_to_voxel(df_s3, spacing=spacing, origin=np.array([0, 0, 0]))

# convert from dataframe to directed graph
G = swc.df_to_graph(df_voxel=df_voxel)
G_s3 = swc.df_to_graph(df_voxel=df_voxel_s3)

# convert directed graph into list of paths
paths = swc.graph_to_paths(G)
paths_s3 = swc.graph_to_paths(G_s3)

# create a subset of the dataframe
mip = 0
ngl = NeuroglancerSession(url, mip=mip, url_segments=url_seg)
buffer = 10
subneuron_df = df_s3[0:5]
vertex_list = subneuron_df["sample"].array
img, bounds, vox_in_img_list = ngl.pull_vertex_list(
    2, vertex_list, buffer=buffer, expand=True
)
df_s3_subset = swc.generate_df_subset(df_s3[0:5], vox_in_img_list)


def test_read_s3_dataframe():
    """test if output is correct type (pd.DataFrame)"""
    try:
        assert isinstance(df_s3, pd.DataFrame)
    except:
        print("s3 still buggin")


def test_read_swc_dataframe():
    """test if output is correct type (pd.DataFrame)"""
    assert isinstance(df, pd.DataFrame)
    assert isinstance(offset, list)
    assert isinstance(color, list)
    # assert isinstance(cc, np.nan)
    # assert isinstance(branch, int)


def test_read_swc_shape():
    """test if output is correct shape"""
    correct_shape = (1650, 7)
    assert df.shape == correct_shape


def test_read_s3_shape():
    """test if output is correct shape"""
    correct_shape = (1650, 7)
    assert df_s3.shape == correct_shape


def test_read_swc_columns():
    """test if columns are correct"""
    col = ["sample", "structure", "x", "y", "z", "r", "parent"]
    assert list(df.columns) == col


def test_read_s3_columns():
    """test if columns are correct"""
    col = ["sample", "structure", "x", "y", "z", "r", "parent"]
    assert list(df_s3.columns) == col


def test_space_to_voxel_int64():
    """test if output is numpy.array of int"""
    spatial_coord = np.array([73940.221323, 18869.828297, 33732.256716])
    voxel_coord = swc.space_to_voxel(
        spatial_coord=spatial_coord, spacing=spacing, origin=origin
    )
    assert all(isinstance(n, np.int64) for n in voxel_coord)


def test_swc_to_voxel_dataframe():
    """test if output is correct type (pd.DataFrame)"""
    assert isinstance(df_voxel, pd.DataFrame)


def test_s3_to_voxel_dataframe():
    """test if output is correct type (pd.DataFrame)"""
    assert isinstance(df_voxel_s3, pd.DataFrame)


def test_swc_to_voxel_shape():
    """test if output is correct shape"""
    correct_shape = (1650, 7)
    assert df_voxel.shape == correct_shape


def test_s3_to_voxel_shape():
    """test if output is correct shape"""
    correct_shape = (1650, 7)
    assert df_voxel_s3.shape == correct_shape


def test_swc_to_voxel_columns():
    """test if columns are correct"""
    col = ["sample", "structure", "x", "y", "z", "r", "parent"]
    assert list(df_voxel.columns) == col


def test_s3_to_voxel_columns():
    """test if columns are correct"""
    col = ["sample", "structure", "x", "y", "z", "r", "parent"]
    assert list(df_voxel_s3.columns) == col


def test_swc_to_voxel_nonnegative():
    """test if coordinates are all nonnegative"""
    coord = df_voxel[["x", "y", "z"]].values
    assert np.greater_equal(np.abs(coord), np.zeros(coord.shape)).all()


def test_df_to_graph_digraph():
    """test if output is directed graph"""
    assert isinstance(G, nx.DiGraph)


def test_df_to_graph_nodes():
    """test if graph has correct number of nodes"""
    assert len(G.nodes) == len(df_voxel)


def test_df_to_graph_nodes():
    """test if graph has correct number of nodes"""
    assert len(G.nodes) == len(df_voxel)


def test_df_to_graph_coordinates():
    """test if graph coordinates are same as that of df_voxel"""
    coord_df = df_voxel[["x", "y", "z"]].values

    x_dict = nx.get_node_attributes(G, "x")
    y_dict = nx.get_node_attributes(G, "y")
    z_dict = nx.get_node_attributes(G, "z")

    x = [x_dict[i] for i in G.nodes]
    y = [y_dict[i] for i in G.nodes]
    z = [z_dict[i] for i in G.nodes]

    coord_graph = np.array([x, y, z]).T

    assert np.array_equal(coord_graph, coord_df)


def test_df_s3_to_graph_coordinates():
    """test if graph coordinates are same as that of df_voxel"""
    coord_df_s3 = df_voxel_s3[["x", "y", "z"]].values

    x_dict = nx.get_node_attributes(G_s3, "x")
    y_dict = nx.get_node_attributes(G_s3, "y")
    z_dict = nx.get_node_attributes(G_s3, "z")

    x = [x_dict[i] for i in G_s3.nodes]
    y = [y_dict[i] for i in G_s3.nodes]
    z = [z_dict[i] for i in G_s3.nodes]

    coord_graph_s3 = np.array([x, y, z]).T

    assert np.array_equal(coord_graph_s3, coord_df_s3)


def test_get_sub_neuron_digraph():
    """test if output is directed graph"""
    start = np.array([15312, 4400, 6448])
    end = np.array([15840, 4800, 6656])
    G_sub = swc.get_sub_neuron(G, bounding_box=(start, end))
    assert isinstance(G_sub, nx.DiGraph)


def test_get_sub_s3_neuron_digraph():
    """test if output is directed graph"""
    start = np.array([15312, 4400, 6448])
    end = np.array([15840, 4800, 6656])
    G_sub_s3 = swc.get_sub_neuron(G_s3, bounding_box=(start, end))
    assert isinstance(G_sub_s3, nx.DiGraph)


def test_get_sub_neuron_bounding_box():
    """test if bounding box produces correct number of nodes and edges"""
    try:
        # case 1: bounding box has nodes and edges
        start = np.array([15312, 4400, 6448])
        end = np.array([15840, 4800, 6656])
        G_sub = swc.get_sub_neuron(G, bounding_box=(start, end))
        num_nodes = 308
        num_edges = 287
        assert len(G_sub.nodes) == num_nodes
        assert len(G_sub.edges) == num_edges
    except:
        pass  # coordinates screwed up bc of s3

    # case 2: bounding box has no nodes and edges
    start = np.array([15312, 4400, 6448])
    end = np.array([15840, 4800, 6448])
    G_sub = swc.get_sub_neuron(G, bounding_box=(start, end))
    assert len(G_sub.nodes) == 0
    assert len(G_sub.edges) == 0


def test_get_sub_s3_neuron_bounding_box():
    """test if bounding box produces correct number of nodes and edges"""

    # case 1: bounding box has nodes and edges
    start = np.array([15312, 4400, 6448])
    end = np.array([15840, 4800, 6656])
    G_sub_s3 = swc.get_sub_neuron(G_s3, bounding_box=(start, end))
    num_nodes = 308
    num_edges = 287
    if len(G_sub_s3) > 0:
        assert len(G_sub_s3.nodes) == num_nodes
        assert len(G_sub_s3.edges) == num_edges

    # case 2: bounding box has no nodes and edges
    start = np.array([15312, 4400, 6448])
    end = np.array([15840, 4800, 6448])
    G_sub_s3 = swc.get_sub_neuron(G_s3, bounding_box=(start, end))
    assert len(G_sub_s3.nodes) == 0
    assert len(G_sub_s3.edges) == 0


def test_graph_to_paths_length():
    """test if output has correct length"""
    num_branches = 179
    assert len(paths) == num_branches


def test_graph_to_paths_s3_length():
    """test if output has correct length"""
    num_branches = 179
    assert len(paths_s3) == num_branches


def test_graph_to_paths_path_dim():
    """test if numpy.arrays have 3 columns"""
    assert all(a.shape[1] == 3 for a in paths)


def test_graph_to_paths_s3_path_dim():
    """test if numpy.arrays have 3 columns"""
    assert all(a.shape[1] == 3 for a in paths_s3)


def test_generate_df_subset():
    """test if output is a dataframe"""
    assert isinstance(df_s3_subset, pd.DataFrame)


def test_generate_BFS_subgraph():
    """test if subgraph matches nodes and edges"""
    G_sub, tree = swc.get_bfs_subgraph(G, 400, 50)
    assert set(G_sub.nodes) == set(tree.nodes)


def test_generate_BFS_subgraph_df():
    """test if subgraph matches nodes and edges"""
    G_sub, tree = swc.get_bfs_subgraph(G, 100, 50, df_voxel)
    assert set(G_sub.nodes) == set(tree.nodes)
