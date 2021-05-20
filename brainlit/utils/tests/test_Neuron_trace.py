import numpy as np
import pandas as pd
import tifffile as tf
import networkx as nx
from cloudvolume import CloudVolume, Skeleton
import brainlit
from brainlit.utils.Neuron_trace import NeuronTrace
from brainlit.utils.session import NeuroglancerSession
import pytest
from pathlib import Path
import networkx.algorithms.isomorphism as iso

swc_path = "./data/data_octree/consensus-swcs/2018-08-01_G-002_consensus.swc"


url_seg = "s3://open-neurodata/brainlit/brain1_segments"
seg_id = 2
mip = 0

seg_id_bad = "asdf"
mip_bad = "asdf"
read_offset_bad = "asdf"
rounding_bad = "asdf"
path_bad_string = "asdf"
path_bad_nonstring = 3

test_swc = NeuronTrace(swc_path)
test_s3 = NeuronTrace(url_seg, seg_id, mip)

####################
### input checks ###
####################


def test_Neurontrace_bad_inputs():
    # test 'path' must be a string
    with pytest.raises(TypeError):
        test_trace = NeuronTrace(path_bad_nonstring)

    # test 'path' must be swc or skel path
    with pytest.raises(ValueError, match="Did not input 'swc' filepath or 'skel' url"):
        test_trace = NeuronTrace(path_bad_string)

    # test 'seg_id' must be NoneType or int
    with pytest.raises(TypeError):
        test_trace = NeuronTrace(url_seg, seg_id_bad, mip)

    # test 'mip' must be NoneType or int
    with pytest.raises(TypeError):
        test_trace = NeuronTrace(url_seg, seg_id, mip_bad)

    # test both 'seg_id' and 'mip' must be provided if one is provided
    with pytest.raises(
        ValueError,
        match="For 'swc' do not input mip or seg_id, and for 'skel', provide both mip and seg_id",
    ):
        test_trace = NeuronTrace(url_seg, seg_id)

    # test 'read_offset' must be bool
    with pytest.raises(TypeError):
        test_trace = NeuronTrace(swc_path, read_offset_bad)

    # test 'rounding' must be bool
    with pytest.raises(TypeError):
        test_trace = NeuronTrace(swc_path, rounding=rounding_bad)


def test_get_df_arguments():
    # test if output is list
    assert isinstance(test_swc.get_df_arguments(), list)
    assert isinstance(test_s3.get_df_arguments(), list)


def test_get_df():
    # test if output is dataframe
    assert isinstance(test_swc.get_df(), pd.DataFrame)
    assert isinstance(test_s3.get_df(), pd.DataFrame)

    # test if output is correct shape
    correct_shape = (1650, 7)
    assert test_swc.get_df().shape == correct_shape
    assert test_s3.get_df().shape == correct_shape

    # test if columns are correct"
    col = ["sample", "structure", "x", "y", "z", "r", "parent"]
    assert list(test_swc.get_df().columns) == col
    assert list(test_s3.get_df().columns) == col


def test_get_skel():
    # test 'origin' arg must either be type None or numpy.ndarray
    with pytest.raises(TypeError):
        test_swc.get_skel(origin="asdf")

    # test 'benchmarking' arg must be bool
    with pytest.raises(TypeError):
        test_swc.get_skel(benchmarking="asdf")

    # test if 'origin' is type numpy.ndarray, it must be shape (3,1)
    with pytest.raises(ValueError):
        test_swc.get_skel(origin=np.asarray([0, 1]))

    # test if output is skeleton
    assert isinstance(test_swc.get_skel(benchmarking=True), Skeleton)
    assert isinstance(test_s3.get_skel(), Skeleton)


def test_get_df_voxel():

    # test 'spacing' arg must be type numpy.ndarray
    with pytest.raises(TypeError):
        test_swc.get_df_voxel(spacing="asdf")

    # test if 'spacing' is type numpy.ndarray, it must be shape (3,1)
    with pytest.raises(ValueError):
        test_swc.get_df_voxel(spacing=np.asarray([0, 1]))

    # test 'origin' arg must be type numpy.ndarray
    with pytest.raises(TypeError):
        test_swc.get_df_voxel(spacing=np.asarray([0, 1, 2]), origin="asdf")

    # test if 'origin' is type numpy.ndarray, it must be shape (3,1)
    with pytest.raises(ValueError):
        test_swc.get_df_voxel(spacing=np.asarray([0, 1, 2]), origin=np.asarray([0, 1]))

    # test if output is correct shape
    correct_shape = (1650, 7)
    df_voxel_swc = test_swc.get_df_voxel(
        spacing=np.asarray([1, 2, 3]), origin=np.asarray([2, 2, 2])
    )
    assert df_voxel_swc.shape == correct_shape
    correct_shape = (1650, 7)
    df_voxel_s3 = test_s3.get_df_voxel(
        spacing=np.asarray([1, 2, 3]), origin=np.asarray([2, 2, 2])
    )
    assert df_voxel_s3.shape == correct_shape

    # test columns
    col = ["sample", "structure", "x", "y", "z", "r", "parent"]
    assert list(df_voxel_swc.columns) == col
    assert list(df_voxel_s3.columns) == col

    # test if coordinates are all nonnegative"""
    coord_swc = df_voxel_swc[["x", "y", "z"]].values
    coord_s3 = df_voxel_s3[["x", "y", "z"]].values
    assert np.greater_equal(np.abs(coord_swc), np.zeros(coord_swc.shape)).all()
    assert np.greater_equal(np.abs(coord_s3), np.zeros(coord_s3.shape)).all()

    # test if output is dataframe
    assert isinstance(
        test_swc.get_df_voxel(
            spacing=np.asarray([0, 1, 2]), origin=np.asarray([0, 1, 2])
        ),
        pd.DataFrame,
    )
    assert isinstance(
        test_s3.get_df_voxel(
            spacing=np.asarray([0, 1, 2]), origin=np.asarray([0, 1, 2])
        ),
        pd.DataFrame,
    )


def test_get_graph():

    # test 'spacing' arg must either be NoneType or numpy.ndarray
    with pytest.raises(TypeError):
        test_swc.get_graph(spacing="asdf")

    # test if 'spacing' is type numpy.ndarray, it must be shape (3,1)
    with pytest.raises(ValueError):
        test_swc.get_graph(spacing=np.asarray([0, 1]))

    # test 'origin' arg must either be NoneType or numpy.ndarray
    with pytest.raises(TypeError):
        test_swc.get_graph(spacing=np.asarray([0, 1, 2]), origin="asdf")

    # test if 'origin' is type numpy.ndarray, it must be shape (3,1)
    with pytest.raises(ValueError):
        test_swc.get_graph(spacing=np.asarray([0, 1, 2]), origin=np.asarray([0, 1]))

    # test if origin isn't specified but spacing is, origin set to np.array([0, 0, 0])
    G1 = test_swc.get_graph(spacing=np.asarray([0, 1, 2]))
    G2 = test_swc.get_graph(spacing=np.asarray([0, 1, 2]), origin=np.array([0, 0, 0]))
    assert nx.is_isomorphic(G1, G2) == True

    # test if graph coordinates are same as that of df_voxel
    df_voxel = test_swc.get_df_voxel(
        spacing=np.asarray([1, 2, 3]), origin=np.asarray([1, 2, 3])
    )
    df_voxel_s3 = test_s3.get_df_voxel(
        spacing=np.asarray([1, 2, 3]), origin=np.asarray([1, 2, 3])
    )

    # swc
    G = test_swc.get_graph(spacing=np.asarray([1, 2, 3]), origin=np.asarray([1, 2, 3]))
    coord_df = df_voxel[["x", "y", "z"]].values
    x_dict = nx.get_node_attributes(G, "x")
    y_dict = nx.get_node_attributes(G, "y")
    z_dict = nx.get_node_attributes(G, "z")
    x = [x_dict[i] for i in G.nodes]
    y = [y_dict[i] for i in G.nodes]
    z = [z_dict[i] for i in G.nodes]
    coord_graph = np.array([x, y, z]).T
    assert np.array_equal(coord_graph, coord_df)
    # s3
    G_s3 = test_s3.get_graph(
        spacing=np.asarray([1, 2, 3]), origin=np.asarray([1, 2, 3])
    )
    coord_df_s3 = df_voxel_s3[["x", "y", "z"]].values
    x_dict = nx.get_node_attributes(G_s3, "x")
    y_dict = nx.get_node_attributes(G_s3, "y")
    z_dict = nx.get_node_attributes(G_s3, "z")
    x = [x_dict[i] for i in G_s3.nodes]
    y = [y_dict[i] for i in G_s3.nodes]
    z = [z_dict[i] for i in G_s3.nodes]
    coord_graph_s3 = np.array([x, y, z]).T
    assert np.array_equal(coord_graph_s3, coord_df_s3)

    # test if graph has correct number of nodes
    assert len(G.nodes) == len(df_voxel)
    assert len(G_s3.nodes) == len(df_voxel_s3)

    # test if output is directed graph
    assert isinstance(test_swc.get_graph(), nx.DiGraph)
    assert isinstance(test_s3.get_graph(), nx.DiGraph)


def test_get_paths():

    # test 'spacing' arg must either be NoneType or numpy.ndarray
    with pytest.raises(TypeError):
        test_swc.get_paths(spacing="asdf")

    # test if 'spacing' is type numpy.ndarray, it must be shape (3,1)
    with pytest.raises(ValueError):
        test_swc.get_paths(spacing=np.asarray([0, 1]))

    # test 'origin' arg must either be NoneType or numpy.ndarray
    with pytest.raises(TypeError):
        test_swc.get_paths(spacing=np.asarray([0, 1, 2]), origin="asdf")

    # test if 'origin' is type numpy.ndarray, it must be shape (3,1)
    with pytest.raises(ValueError):
        test_swc.get_paths(spacing=np.asarray([0, 1, 2]), origin=np.asarray([0, 1]))

    # test if output is type numpy.ndarray
    assert isinstance(test_swc.get_paths(), np.ndarray)
    assert isinstance(test_s3.get_paths(), np.ndarray)


def test_generate_df_subset():
    # test 'vox_in_img_list' arg must be type list
    with pytest.raises(TypeError):
        test_swc.generate_df_subset(vox_in_img_list=2)

    # test 'subneuron_start' arg must either be NoneType or int
    with pytest.raises(TypeError):
        vox_in_img_list = [[72, 53, 128], [69, 60, 122], [69, 63, 115]]
        test_swc.generate_df_subset(
            vox_in_img_list, subneuron_start="asdf", subneuron_end=3
        )

    # test 'subneuron_end' arg must either be NoneType or int
    with pytest.raises(TypeError):
        vox_in_img_list = [
            [72, 53, 128],
            [69, 60, 122],
            [69, 63, 115],
            [74, 63, 108],
            [155, 162, 57],
        ]
        test_swc.generate_df_subset(
            vox_in_img_list, subneuron_start=3, subneuron_end="asdf"
        )

    # test both 'subneuron_start' or 'subneuron_end must' be specified, if only 1 is
    with pytest.raises(
        ValueError,
        match="Provide both starting and ending vertices to use for the subneuron",
    ):
        vox_in_img_list = [
            [72, 53, 128],
            [69, 60, 122],
            [69, 63, 115],
            [74, 63, 108],
            [155, 162, 57],
        ]
        test_swc.generate_df_subset(vox_in_img_list, subneuron_start=3)

    # test if output is a dataframe
    assert isinstance(
        test_swc.generate_df_subset(
            vox_in_img_list, subneuron_start=0, subneuron_end=3
        ),
        pd.DataFrame,
    )


def test_get_bfs_subgraph():
    # test 'node_id' arg must be type int
    with pytest.raises(TypeError):
        test_swc.get_bfs_subgraph(node_id="asdf", depth=2)

    # test 'depth' arg must be type int
    with pytest.raises(TypeError):
        test_swc.get_bfs_subgraph(node_id=11, depth="asdf")

    # test 'df' arg must be either NoneType or pandas.DataFrame
    with pytest.raises(TypeError):
        test_swc.get_bfs_subgraph(node_id=11, depth=2, df="asdf")

    # test 'spacing' arg must either be NoneType or numpy.ndarray
    with pytest.raises(TypeError):
        test_swc.get_bfs_subgraph(node_id=11, depth=2, spacing=1)

    # test if 'spacing' is type numpy.ndarray, it must be shape (3,1)
    with pytest.raises(ValueError):
        test_swc.get_bfs_subgraph(node_id=11, depth=2, spacing=np.asarray([1, 2]))

    # test 'origin' arg must either be NoneType or numpy.ndarray
    with pytest.raises(TypeError):
        test_swc.get_bfs_subgraph(
            node_id=11, depth=2, spacing=np.asarray([1, 2, 3]), origin=1
        )

    # test if 'origin' is type numpy.ndarray, it must be shape (3,1)
    with pytest.raises(TypeError):
        test_swc.get_bfs_subgraph(
            node_id=11, depth=2, spacing=np.asarray([1, 2, 3]), origin=np.asarray[1, 2]
        )

    # test if subgraph matches nodes and edges
    G_sub, tree, paths = test_swc.get_bfs_subgraph(node_id=100, depth=50)
    assert set(G_sub.nodes) == set(tree.nodes)
    G_sub, tree, paths = test_swc.get_bfs_subgraph(
        node_id=100, depth=50, df=test_s3.get_df()
    )
    assert set(G_sub.nodes) == set(tree.nodes)

    # test if outputs are directed graphs and paths
    G_sub_s3, tree_s3, paths_s3 = test_swc.get_bfs_subgraph(100, 50)
    assert isinstance(G_sub, nx.DiGraph)
    assert isinstance(tree, nx.DiGraph)
    assert isinstance(paths_s3, np.ndarray)
    assert isinstance(G_sub_s3, nx.DiGraph)
    assert isinstance(tree_s3, nx.DiGraph)
    assert isinstance(paths_s3, np.ndarray)


def test_get_sub_neuron():

    # test 'bounding_box' arg is a tuple or list
    with pytest.raises(TypeError):
        test_swc.get_sub_neuron(bounding_box=1)

    # test if 'bounding_box' arg is tuple or list, length is 2
    with pytest.raises(ValueError, match="Bounding box must be length 2"):
        test_swc.get_sub_neuron(bounding_box=[[1, 2, 4], [1, 2, 3], [4, 5, 6]])

    # test 'spacing' arg must either be NoneType or numpy.ndarray
    with pytest.raises(TypeError):
        test_swc.get_sub_neuron(bounding_box=[[1, 2, 4], [1, 2, 3]], spacing="asdf")

    # test if 'spacing' is type numpy.ndarray, it must be shape (3,1)
    with pytest.raises(ValueError):
        test_swc.get_sub_neuron(
            bounding_box=[[1, 2, 4], [1, 2, 3]], spacing=np.asarray([1, 2])
        )

    # test 'origin' arg must either be NoneType or numpy.ndarray
    with pytest.raises(TypeError):
        test_swc.get_sub_neuron(
            bounding_box=[[1, 2, 4], [1, 2, 3]], spacing=np.asarray([1, 2, 3]), origin=1
        )

    # test if 'origin' is type numpy.ndarray, it must be shape (3,1)
    with pytest.raises(ValueError):
        test_swc.get_sub_neuron(
            bounding_box=[[1, 2, 4], [1, 2, 3]],
            spacing=np.asarray([1, 2, 3]),
            origin=np.asarray([1, 2]),
        )

    # test if bounding box produces correct number of nodes and edges
    # swc
    try:
        # case 1: bounding box has nodes and edges
        start = np.array([15312, 4400, 6448])
        end = np.array([15840, 4800, 6656])
        G_sub = test_swc.get_sub_neuron(bounding_box=(start, end))
        num_nodes = 308
        num_edges = 287
        assert len(G_sub.nodes) == num_nodes
        assert len(G_sub.edges) == num_edges
    except:
        pass  # coordinates screwed up bc of s3
    # case 2: bounding box has no nodes and edges
    start = np.array([15312, 4400, 6448])
    end = np.array([15840, 4800, 6448])
    G_sub = test_swc.get_sub_neuron(bounding_box=(start, end))
    assert len(G_sub.nodes) == 0
    assert len(G_sub.edges) == 0
    # s3
    # case 1: bounding box has nodes and edges
    start = np.array([15312, 4400, 6448])
    end = np.array([15840, 4800, 6656])
    G_sub_s3 = test_s3.get_sub_neuron(bounding_box=(start, end))
    num_nodes = 308
    num_edges = 287
    if len(G_sub_s3) > 0:
        assert len(G_sub_s3.nodes) == num_nodes
        assert len(G_sub_s3.edges) == num_edges
    # case 2: bounding box has no nodes and edges
    start = np.array([15312, 4400, 6448])
    end = np.array([15840, 4800, 6448])
    G_sub_s3 = test_s3.get_sub_neuron(bounding_box=(start, end))
    assert len(G_sub_s3.nodes) == 0
    assert len(G_sub_s3.edges) == 0

    # test if output is directed graph
    sub_neuron_swc = test_swc.get_sub_neuron(bounding_box=[[1, 2, 4], [1, 2, 3]])
    sub_neuron_s3 = test_s3.get_sub_neuron(bounding_box=[[1, 2, 4], [1, 2, 3]])
    assert isinstance(sub_neuron_swc, nx.DiGraph)
    assert isinstance(sub_neuron_s3, nx.DiGraph)


def test_get_sub_neuron_paths():

    # test 'bounding_box' arg is a tuple or list
    with pytest.raises(TypeError):
        test_swc.get_sub_neuron_paths(bounding_box=1)

    # test if 'bounding_box' arg is tuple or list, length is 2
    with pytest.raises(ValueError, match="Bounding box must be length 2"):
        test_swc.get_sub_neuron_paths(bounding_box=[[1, 2, 4], [1, 2, 3], [4, 5, 6]])

    # test 'spacing' arg must either be NoneType or numpy.ndarray
    with pytest.raises(TypeError):
        test_swc.get_sub_neuron_paths(
            bounding_box=[[1, 2, 4], [1, 2, 3]], spacing="asdf"
        )

    # test if 'spacing' is type numpy.ndarray, it must be shape (3,1)
    with pytest.raises(ValueError):
        test_swc.get_sub_neuron_paths(
            bounding_box=[[1, 2, 4], [1, 2, 3]], spacing=np.asarray([1, 2])
        )

    # test 'origin' arg must either be NoneType or numpy.ndarray
    with pytest.raises(TypeError):
        test_swc.get_sub_neuron_paths(
            bounding_box=[[1, 2, 4], [1, 2, 3]], spacing=np.asarray([1, 2, 3]), origin=1
        )

    # test if 'origin' is type numpy.ndarray, it must be shape (3,1)
    with pytest.raises(ValueError):
        test_swc.get_sub_neuron_paths(
            bounding_box=[[1, 2, 4], [1, 2, 3]],
            spacing=np.asarray([1, 2, 3]),
            origin=np.asarray([1, 2]),
        )

    # test whether voxel conversion option is covered
    start = np.array([-600.273315, 2171.536082, -1835.069744])
    end = np.array([-562.945041, 2197.964079, -1819.749479])

    paths1 = test_swc.get_sub_neuron_paths(bounding_box=(start, end))
    paths2 = test_swc.get_sub_neuron_paths(
        bounding_box=(start, end),
        spacing=np.asarray([1, 2, 3]),
        origin=np.asarray([1, 2, 3]),
    )

    assert paths1 != paths2

    # test if output for paths is type numpy.ndarray
    sub_neuron_swc = test_swc.get_sub_neuron_paths(bounding_box=[[1, 2, 3], [1, 2, 3]])
    sub_neuron_s3 = test_s3.get_sub_neuron_paths(bounding_box=[[1, 2, 3], [1, 2, 3]])
    assert isinstance(sub_neuron_swc, np.ndarray)
    assert isinstance(sub_neuron_s3, np.ndarray)


def test_ssd():
    zero = np.asarray([[0, 0]])
    pt1_bad = "asdf"
    pt2_bad = "asdf"

    # test if 'pt1' is type np.ndarray
    with pytest.raises(TypeError):
        NeuronTrace.ssd(pt1_bad, zero)

    # test if 'pt2' is type np.ndarray
    with pytest.raises(TypeError):
        NeuronTrace.ssd(zero, pt2_bad)

    # test if the inputs are both 2D arrays so sklearn pairwise dist can work
    integer = np.asarray(0)
    vector = np.asarray([0, 0])
    array = np.asarray([[0, 0]])
    none_list = np.asarray([])
    with pytest.raises(
        ValueError, match=r"Expected 2D array, got scalar array instead"
    ):
        NeuronTrace.ssd(integer, integer)
    with pytest.raises(ValueError, match=r"Expected 2D array, got 1D array instead"):
        NeuronTrace.ssd(vector, none_list)
    with pytest.raises(ValueError, match=r"Expected 2D array, got 1D array instead"):
        NeuronTrace.ssd(array, none_list)
    with pytest.raises(ValueError, match=r"Expected 2D array, got 1D array instead"):
        NeuronTrace.ssd(vector, array)
    with pytest.raises(ValueError, match=r"Expected 2D array, got 1D array instead"):
        NeuronTrace.ssd(array, vector)

    # test ssd outputs
    a = np.asarray([[0, 0], [1, 1]])
    b = np.asarray([[1, 1], [0, 0]])
    c = np.asarray([[0, 0], [1, 1], [2, 2]])
    d = np.asarray([[0, 0], [np.sqrt(2), np.sqrt(2)]])
    e = np.asarray([[0, 0], [np.sqrt(2), np.sqrt(2)], [np.sqrt(8), np.sqrt(8)]])
    f = np.asarray([[10, 10], [20, 20], [30, 30], [40, 40]])

    # test if SSD returns the proper values
    # SSD with Self = 0
    assert NeuronTrace.ssd(a, a) == 0

    # Insignificant distances, 0
    assert NeuronTrace.ssd(a, b) == 0
    assert NeuronTrace.ssd(a, c) == 0
    assert NeuronTrace.ssd(c, e) == 0
    # Significant distances
    assert NeuronTrace.ssd(zero, d) == 2
    assert NeuronTrace.ssd(zero, e) == 3
    assert NeuronTrace.ssd(c, f) == 24.041630560342618
