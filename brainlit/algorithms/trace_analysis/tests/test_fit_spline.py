import pytest
from brainlit.algorithms.trace_analysis.fit_spline import GeometricGraph
from brainlit.utils.Neuron_trace import NeuronTrace
import networkx as nx
from scipy.interpolate import splprep
import numpy as np
from pathlib import Path
import pandas as pd

top_level = Path(__file__).parents[4] / "data"
input = (top_level / "data_octree").as_posix()
url = (top_level / "test_upload").as_uri()
url_seg = url + "_segments"
url = url + "/serial"


##############
### inputs ###
##############


def test_fit_spline_tree_invariant_bad_input():

    # test nodes must have a 'loc' attribute
    neuron_no_loc = GeometricGraph()
    neuron_no_loc.add_node(1)
    neuron_no_loc.add_node(2, loc=np.array([100, 100, 200]))
    # build spline tree using fit_spline_tree_invariant()
    with pytest.raises(KeyError, match=r"some nodes are missing the 'loc' attribute"):
        neuron_no_loc.fit_spline_tree_invariant()

    # test 'loc' attribute must be numpy.ndarray
    neuron_wrong_loc_type = GeometricGraph()
    neuron_wrong_loc_type.add_node(1, loc={})
    neuron_wrong_loc_type.add_node(2, loc=np.array([100, 100, 200]))
    with pytest.raises(
        TypeError, match=r"{} should be <class 'numpy.ndarray'>, not <class 'dict'>."
    ):
        neuron_wrong_loc_type.fit_spline_tree_invariant()

    # test 'loc' attribute must be a flat array
    neuron_nested_loc = GeometricGraph()
    neuron_nested_loc.add_node(1, loc=np.array([[]]))
    neuron_nested_loc.add_node(2, loc=np.array([100, 100, 200]))
    with pytest.raises(ValueError, match=r"nodes must be flat arrays"):
        neuron_nested_loc.fit_spline_tree_invariant()

    # test 'loc' attribute cannot be empty
    neuron_empty_loc = GeometricGraph()
    neuron_empty_loc.add_node(1, loc=np.array([]))
    neuron_empty_loc.add_node(2, loc=np.array([100, 100, 200]))
    with pytest.raises(ValueError, match=r"nodes cannot have empty 'loc' attributes"):
        neuron_empty_loc.fit_spline_tree_invariant()

    # test 'loc' attribute must be real-valued
    neuron_non_real_valued_loc = GeometricGraph()
    neuron_non_real_valued_loc.add_node(1, loc=np.array(["a"]))
    neuron_non_real_valued_loc.add_node(2, loc=np.array([100, 100, 200]))
    with pytest.raises(
        TypeError,
        match=r"\['a'\] elements should be \(<class 'numpy.integer'>, <class 'float'>\).",
    ):
        neuron_non_real_valued_loc.fit_spline_tree_invariant()

    # test 'loc' attribute must have 3 coordinates
    neuron_wrong_coordinates = GeometricGraph()
    neuron_wrong_coordinates.add_node(1, loc=np.array([1, 2]))
    neuron_wrong_coordinates.add_node(2, loc=np.array([100, 100, 200]))
    with pytest.raises(ValueError, match=r"'loc' attributes must contain 3 coordinate"):
        neuron_wrong_coordinates.fit_spline_tree_invariant()

    # test 'loc' attributes must be unique
    neuron_duplicate_loc = GeometricGraph()
    neuron_duplicate_loc.add_node(1, loc=np.array([100, 100, 200]))
    neuron_duplicate_loc.add_node(2, loc=np.array([100, 100, 200]))
    with pytest.raises(ValueError, match=r"there are duplicate nodes"):
        neuron_duplicate_loc.fit_spline_tree_invariant()

    # test edges must be a valid cover of the graph
    neuron_no_edges = GeometricGraph()
    neuron_no_edges.add_node(1, loc=np.array([100, 100, 200]))
    neuron_no_edges.add_node(2, loc=np.array([200, 200, 400]))
    with pytest.raises(
        ValueError, match=r"the edges are not a valid cover of the graph"
    ):
        neuron_no_edges.fit_spline_tree_invariant()

    # test there cannot be undirected cycles in the graph
    neuron_cycles = GeometricGraph()
    neuron_cycles.add_node(1, loc=np.array([100, 100, 200]))
    neuron_cycles.add_node(2, loc=np.array([200, 0, 200]))
    neuron_cycles.add_node(3, loc=np.array([200, 300, 200]))
    neuron_cycles.add_node(4, loc=np.array([300, 400, 200]))
    neuron_cycles.add_node(5, loc=np.array([100, 500, 200]))
    # add edges
    neuron_cycles.add_edge(2, 1)
    neuron_cycles.add_edge(3, 2)
    neuron_cycles.add_edge(4, 3)
    neuron_cycles.add_edge(5, 4)
    neuron_cycles.add_edge(3, 5)
    with pytest.raises(ValueError, match=r"the graph contains undirected cycles"):
        neuron_cycles.fit_spline_tree_invariant()

    # test there cannot be disconnected segments in the graph
    neuron_disconnected_segments = GeometricGraph()
    neuron_disconnected_segments.add_node(1, loc=np.array([100, 100, 200]))
    neuron_disconnected_segments.add_node(2, loc=np.array([200, 0, 200]))
    neuron_disconnected_segments.add_node(3, loc=np.array([200, 300, 200]))
    neuron_disconnected_segments.add_node(4, loc=np.array([300, 400, 200]))
    neuron_disconnected_segments.add_node(5, loc=np.array([100, 500, 200]))
    # add edges
    neuron_disconnected_segments.add_edge(2, 1)
    neuron_disconnected_segments.add_edge(3, 4)
    neuron_disconnected_segments.add_edge(3, 5)
    # build spline tree using fit_spline_tree_invariant()
    with pytest.raises(ValueError, match=r"the graph contains disconnected segments"):
        neuron_disconnected_segments.fit_spline_tree_invariant()


def test_init_from_bad_df():
    d = {
        "sample": [1, 2],
        "structure": [0, 0],
        "x": [1, 1],
        "y": [2, 2],
        "z": [3, 3],
        "r": [1, 1],
        "parent": [-1, 1],
    }
    df = pd.DataFrame(data=d)
    print(df)
    with pytest.raises(
        ValueError, match="cannot build GeometricGraph with duplicate nodes"
    ):
        GeometricGraph(df=df)


##################
### validation ###
##################


def test_init_from_df():
    s3_trace = NeuronTrace(path=url_seg, seg_id=2, mip=0, rounding=False)
    df_s3 = s3_trace.get_df()
    G = GeometricGraph(df=df_s3)
    assert isinstance(G, GeometricGraph)


def test_splNum():

    # test the number of splines is correct
    neuron = GeometricGraph()
    # add nodes
    neuron.add_node(1, loc=np.array([100, 100, 200]))
    neuron.add_node(2, loc=np.array([200, 0, 200]))
    neuron.add_node(3, loc=np.array([200, 300, 200]))
    neuron.add_node(4, loc=np.array([300, 400, 200]))
    neuron.add_node(5, loc=np.array([100, 500, 200]))
    neuron.add_node(6, loc=np.array([300, 500, 200]))
    # add edges
    neuron.add_edge(2, 1)
    neuron.add_edge(2, 3)
    neuron.add_edge(3, 4)
    neuron.add_edge(3, 5)
    neuron.add_edge(4, 6)
    spline_tree = neuron.fit_spline_tree_invariant()
    # expect to have 2 splines
    assert len(spline_tree.nodes) == 2


def test_CompareLen():

    # test when there exists one longest path
    neuron_long1 = GeometricGraph()
    # add nodes
    neuron_long1.add_node(1, loc=np.array([100, 0, 200]))
    neuron_long1.add_node(2, loc=np.array([100, 100, 200]))
    neuron_long1.add_node(3, loc=np.array([0, 200, 200]))
    neuron_long1.add_node(4, loc=np.array([200, 300, 200]))
    # add edges
    neuron_long1.add_edge(1, 2)
    neuron_long1.add_edge(2, 3)
    neuron_long1.add_edge(2, 4)
    spline_tree = neuron_long1.fit_spline_tree_invariant()
    # collect all the paths in `PATHS`
    PATHS = []
    for node in spline_tree.nodes:
        PATHS.append(spline_tree.nodes[node]["path"])
    # check if node 4 is in the first spline
    assert 4 in PATHS[0]

    # test when there are multiple equally long paths
    neuron_long4 = GeometricGraph()
    # add nodes
    neuron_long4.add_node(1, loc=np.array([0, -100, 200]))
    neuron_long4.add_node(2, loc=np.array([0, 0, 200]))
    neuron_long4.add_node(4, loc=np.array([100, 100, 200]))
    neuron_long4.add_node(3, loc=np.array([-100, -100, 200]))
    neuron_long4.add_node(6, loc=np.array([100, -100, 200]))
    neuron_long4.add_node(5, loc=np.array([-100, 100, 200]))
    # add edges
    neuron_long4.add_edge(1, 2)
    neuron_long4.add_edge(2, 5)
    neuron_long4.add_edge(2, 6)
    neuron_long4.add_edge(2, 4)
    neuron_long4.add_edge(2, 3)
    spline_tree = neuron_long4.fit_spline_tree_invariant()
    # collect all the paths in `PATHS`
    PATHS = []
    for node in spline_tree.nodes:
        PATHS.append(spline_tree.nodes[node]["path"])
    # check: except the first spline (first edge is added first), all the equal-length splines are added according to the reverse order of the edge addition
    assert 5 in PATHS[0]
    assert 3 in PATHS[1]
    assert 4 in PATHS[2]
    assert 6 in PATHS[3]


def test_spline():

    # Compare the spline parameters u and tck from `fit_spline_tree_invariant` and directly from `scipy.interpolate.splprep`
    neuron = GeometricGraph()
    # add nodes
    neuron.add_node(1, loc=np.array([100, 0, 200]))
    neuron.add_node(2, loc=np.array([100, 100, 200]))
    neuron.add_node(3, loc=np.array([0, 200, 200]))
    neuron.add_node(4, loc=np.array([200, 300, 200]))
    # add edges
    neuron.add_edge(1, 2)
    neuron.add_edge(2, 3)
    neuron.add_edge(2, 4)
    # first path parameters created by `splprep`
    path = [1, 2, 4]
    x = np.zeros((len(path), 3))
    for row, node in enumerate(path):
        x[row, :] = neuron.nodes[node]["loc"]
    m = x.shape[0]
    diffs = np.diff(x, axis=0)
    diffs = np.linalg.norm(diffs, axis=1)
    diffs = np.cumsum(diffs)
    diffs = np.concatenate(([0], diffs))
    k = np.amin([m - 1, 5])
    tck_scipy, u_scipy = splprep([x[:, 0], x[:, 1], x[:, 2]], u=diffs, k=k)
    # first path created by `fit_spline_tree_invariant`
    spline_tree = neuron.fit_spline_tree_invariant()
    spline = spline_tree.nodes[0]["spline"]
    u_fit = spline[1]
    tck_fit = spline[0]
    for n in range(0, len(tck_scipy), 1):
        np.testing.assert_array_equal(tck_scipy[n], tck_fit[n])
    np.testing.assert_array_equal(u_scipy, u_fit)
