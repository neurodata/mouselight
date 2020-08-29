import numpy as np
from pathlib import Path
import re
import pandas as pd
import networkx as nx
from cloudvolume import CloudVolume, Skeleton
from io import StringIO


def read_swc(path):
    """Read a single swc file

    Arguments:
        path {string} -- path to file
        raw {bool} -- whether you are passing the file directly

    Returns:
        df {pandas dataframe} -- indices, coordinates, and parents of each node
        offset {list of floats} -- offset value of fragment
        color {list of ints} -- color
        cc {int} -- cc value, from file name
        branch {int} -- branch number, from file name
    """

    # check input
    file = open(path, "r")
    in_header = True
    offset_found = False
    header_length = -1
    offset = np.nan
    color = np.nan
    cc = np.nan
    branch = np.nan
    while in_header:
        line = file.readline().split()
        if "OFFSET" in line:
            offset_found = True
            idx = line.index("OFFSET") + 1
            offset = [float(line[i]) for i in np.arange(idx, idx + 3)]
        elif "COLOR" in line:
            idx = line.index("COLOR") + 1
            line = line[idx]
            line = line.split(",")
            color = [float(line[i]) for i in np.arange(len(line))]
        elif "NAME" in line:
            idx = line.index("NAME") + 1
            name = line[idx]
            name = re.split(r"_|-|\.", name)
            try:
                idx = name.index("cc") + 1
                cc = int(name[idx])
                idx = name.index("branch") + 1
                branch = int(name[idx])
            except ValueError:
                pass
        elif line[0] != "#":
            in_header = False
        header_length += 1

    if not offset_found:
        raise IOError("No offset information found in: " + path)
    # read coordinates
    df = pd.read_table(
        path,
        names=["sample", "structure", "x", "y", "z", "r", "parent"],
        skiprows=header_length,
        delimiter=" ",
    )
    return df, offset, color, cc, branch


def read_swc_offset(path):
    df, offset, color, cc, branch = read_swc(path)
    df["x"] = df["x"] + offset[0]
    df["y"] = df["y"] + offset[1]
    df["z"] = df["z"] + offset[2]

    return df, color, cc, branch


def append_df(cumulative, new):
    """Append the dataframe of one fragment to the dataframe of one or more other fragments

    Arguments:
        cumulative {pandas dataframe} -- other fragments
        new {pandas dataframe} -- fragment to be appended


    Returns:
        datafranme -- appended result
    """
    # check cumulative df
    samples = cumulative["sample"].values
    unq = np.unique(samples)
    if len(unq) != cumulative.shape[0]:
        msg = (
            "cumulative df has "
            + str(cumulative.shape[0])
            + " rows but only "
            + str(len(unq))
            + " unique sample labels."
        )
        raise ValueError(msg)
    mx = np.amax(unq)
    sample_new = np.arange(mx + 1, mx + 1 + new.shape[0])
    sample_new = np.append(sample_new, -1)
    sample_old = new["sample"].values
    sample_old = np.append(sample_old, -1)
    parent_old = new["parent"].values

    idxs = [
        np.argwhere(sample_old == parent_old[i])[0, 0]
        for i in np.arange(len(parent_old))
    ]
    parent_new = [sample_new[idx] for idx in idxs]

    new["sample"] = sample_new[:-1]
    new["parent"] = parent_new
    cumulative = cumulative.append(new)
    return cumulative.reset_index(drop=True)


def read_swc_dir(path, nFiles=1):
    """Read all swc files in a directory and create a large dataframe of all the results

    Arguments:
        path {string} -- path to directory of swc files

    Keyword Arguments:
        nFiles {int} -- number of files to process in the directory (default: {1})

    Returns:
        pandas dataframe -- dataframe of all the fragment data
    """
    files = list(Path(path).glob("**/*.swc"))[0:nFiles]

    for i, file in enumerate(files):
        df, offset, color, cc, branch = read_swc(file)
        df["x"] = df["x"] + offset[0]
        df["y"] = df["y"] + offset[1]
        df["z"] = df["z"] + offset[2]

        df["color0"] = color[0]
        df["color1"] = color[1]
        df["color2"] = color[2]

        df["cc"] = cc
        df["branch"] = branch

        if i == 0:
            df_cumulative = df
        else:
            df_cumulative = append_df(df_cumulative, df)

    return df_cumulative


def bbox_vox(df):
    """Returns the coordinate of a box center and its dimensions that contain fragments

    Arguments:
        df {pandas dataframe} -- dataframe of points as outputted from read_swc_offset

    Returns:
        start {3 list} -- starting spatial coordinate
        end {3 list} -- ending spatial coordinate
    """
    min_x, max_x = np.min(df.x), np.max(df.x)
    min_y, max_y = np.min(df.y), np.max(df.y)
    min_z, max_z = np.min(df.z), np.max(df.z)

    start = [min_x, min_y, min_z]
    end = [max_x, max_y, max_z]
    return start, end


def read_s3(s3_path, seg_id, mip):
    """Read a s3 bucket path to a skeleton object
    into a pandas dataframe.

    Parameters
    ----------
    s3_path : str
        String representing the path to the s3 bucket
    seg_id : int
        The segement number to pull
    mip : int
        The resolution to use for scaling

    Returns
    -------
    df : :class:`pandas.DataFrame`
        Indicies, coordinates, and parents of each node in the swc.
        Coordinates are in spatial units.
    """
    # TODO check header length

    # check input
    cv = CloudVolume(s3_path, mip=mip)
    skeleton = cv.skeleton.get(seg_id)
    swc_string = skeleton.to_swc()
    string_io = StringIO(swc_string)
    splitted_string = swc_string.split("\n")
    in_h = True
    h_len = -1
    while in_h:
        h_len += 1
        line = splitted_string[h_len]
        if len(line) == 0 or line[0] != "#":
            in_h = False
    df = pd.read_table(
        string_io,
        names=["sample", "structure", "x", "y", "z", "r", "parent"],
        skiprows=h_len,
        sep=" "
        # delim_whitespace=True,
    )
    res = cv.scales[mip]["resolution"]
    df["x"] = np.round(df["x"] / res[0])
    df["y"] = np.round(df["y"] / res[1])
    df["z"] = np.round(df["z"] / res[2])
    return df


def generate_df_subset(swc_df, vox_in_img_list):
    """Read a new subset of swc dataframe in coordinates in img spacing.

    Parameters
    ----------
    swc_df : pd.DataFrame
        DataFrame containing information from swc file
    vox_in_img_list: list
        List of voxels

    Returns
    -------
    df : :class:`pandas.DataFrame`
        Indicies, coordinates (in img spacing) and parents of each node in the swc.
        Coordinates are in spatial units.
    """

    # check input
    df_new = swc_df.copy()
    df_new["x"], df_new["y"], df_new["z"] = (
        vox_in_img_list[:, 0],
        vox_in_img_list[:, 1],
        vox_in_img_list[:, 2],
    )

    return df_new


def space_to_voxel(spatial_coord, spacing, origin=np.array([0, 0, 0])):
    """Converts coordinate from spatial units to voxel units.

    Parameters
    ----------
    spatial_coord : :class:`numpy.array`
        3D coordinate in spatial units. Assumed to be np.array[(x,y,z)]
    spacing : :class:`numpy.array`
        Conversion factor (spatial units/voxel). Assumed to be np.array([x,y,z])
    origin : :class:`numpy.array`
        Origin of the spatial coordinate. Default is (0,0,0). Assumed to be
        np.array([x,y,z])
    Returns
    -------
    voxel_coord : :class:`numpy.array`
        Coordinate in voxel units. Assumed to be np.array([x,y,z])
    """

    voxel_coord = np.round(np.divide(spatial_coord - origin, spacing))
    voxel_coord = voxel_coord.astype(np.int64)
    return voxel_coord


def swc_to_voxel(df, spacing, origin=np.array([0, 0, 0])):
    """Converts coordinates in pd.DataFrame representing swc from spatial units
    to voxel units

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        Indicies, coordinates, and parents of each node in the swc. Coordinates
        are in spatial units.
    spacing : :class:`numpy.array`
        Conversion factor (spatial units/voxel). Assumed to be np.array([x,y,z])
    origin : :class:`numpy.array`
        Origin of the spatial coordinate. Default is (0,0,0). Assumed to be
        np.array([x,y,z])
    Returns
    -------
    df_voxel : :class:`pandas.DataFrame`
        Indicies, coordinates, and parents of each node in the swc. Coordinates
        are in voxel units.
    """
    x = []
    y = []
    z = []
    df_voxel = df.copy()
    for index, row in df_voxel.iterrows():
        vox = space_to_voxel(row[["x", "y", "z"]].to_numpy(), spacing, origin)
        x.append(vox[0])
        y.append(vox[1])
        z.append(vox[2])

    df_voxel["x"] = x
    df_voxel["y"] = y
    df_voxel["z"] = z

    return df_voxel


def df_to_graph(df_voxel):
    """Converts dataframe of swc in voxel coordinates into a directed graph

    Parameters
    ----------
    df_voxel : :class:`pandas.DataFrame`
        Indicies, coordinates, and parents of each node in the swc. Coordinates
        are in voxel units.
    Returns
    -------
    G : :class:`networkx.classes.digraph.DiGraph`
        Neuron from swc represented as directed graph. Coordinates x,y,z are
        node attributes accessed by keys 'x','y','z' respectively.
    """
    G = nx.DiGraph()

    # add nodes
    for index, row in df_voxel.iterrows():
        id = int(row["sample"])

        G.add_node(id)
        G.nodes[id]["x"] = int(row["x"])
        G.nodes[id]["y"] = int(row["y"])
        G.nodes[id]["z"] = int(row["z"])

    # add edges
    for index, row in df_voxel.iterrows():
        child = int(row["sample"])
        parent = int(row["parent"])

        if parent > min(df_voxel["parent"]):
            G.add_edge(parent, child)

    return G


def get_sub_neuron(G, bounding_box):
    """Returns sub-neuron with node coordinates bounded by start and end

    Parameters
    ----------
    G : :class:`networkx.classes.digraph.DiGraph`
        Neuron from swc represented as directed graph. Coordinates x,y,z are
        node attributes accessed by keys 'x','y','z' respectively.
    bounding_box : tuple or list or None
        Defines a bounding box around a sub-region around the neuron. Length 2
        tuple/list. First element is the coordinate of one corner (inclusive) and second element is the coordinate of the opposite corner (exclusive). Both coordinates are numpy.array([x,y,z])in voxel units.
    Returns
    -------
    G_sub : :class:`networkx.classes.digraph.DiGraph`
        Neuron from swc represented as directed graph. Coordinates x,y,z are
        node attributes accessed by keys 'x','y','z' respectively.
    """
    G_sub = G.copy()  # make copy of input G
    start = bounding_box[0]
    end = bounding_box[1]

    # remove nodes that are not neighbors of nodes bounded by start and end
    for node in list(G_sub.nodes):
        neighbors = list(G_sub.successors(node)) + list(G_sub.predecessors(node))

        remove = True

        for id in neighbors + [node]:
            x = G_sub.nodes[id]["x"]
            y = G_sub.nodes[id]["y"]
            z = G_sub.nodes[id]["z"]

            if x >= start[0] and y >= start[1] and z >= start[2]:
                if x < end[0] and y < end[1] and z < end[2]:
                    remove = False

        if remove:
            G_sub.remove_node(node)

    # set origin to start of bounding box
    for id in list(G_sub.nodes):
        G_sub.nodes[id]["x"] = G_sub.nodes[id]["x"] - start[0]
        G_sub.nodes[id]["y"] = G_sub.nodes[id]["y"] - start[1]
        G_sub.nodes[id]["z"] = G_sub.nodes[id]["z"] - start[2]

    return G_sub


def graph_to_paths(G):
    """Converts neuron represented as a directed graph with no cycles into a
    list of paths.

    Parameters
    ----------
    G : :class:`networkx.classes.digraph.DiGraph`
        Neuron from swc represented as directed graph. Coordinates x,y,z are
        node attributes accessed by keys 'x','y','z' respectively.
    Returns
    -------
    paths : list
        List of Nx3 numpy.array. Rows of the array are 3D coordinates in voxel
        units. Each array is one path.
    """
    G_cp = G.copy()  # make copy of input G
    branches = []
    while len(G_cp.edges) != 0:  # iterate over branches
        # get longest branch
        longest = nx.algorithms.dag.dag_longest_path(G_cp)  # list of nodes on the path
        branches.append(longest)

        # remove longest branch
        for idx, e in enumerate(longest):
            if idx < len(longest) - 1:
                G_cp.remove_edge(longest[idx], longest[idx + 1])

    # convert branches into list of paths
    paths = []
    for branch in branches:
        # get vertices in branch as n by 3 numpy.array; n = length of branches
        path = np.zeros((len(branch), 3), dtype=np.int64)
        for idx, node in enumerate(branch):
            path[idx, 0] = np.int64(G_cp.nodes[node]["x"])
            path[idx, 1] = np.int64(G_cp.nodes[node]["y"])
            path[idx, 2] = np.int64(G_cp.nodes[node]["z"])

        paths.append(path)

    return np.array(paths, dtype="object")


def get_bfs_subgraph(G, node_id, depth, df=None):
    """
    Creates a spanning subgraph from a seed node and parent graph using BFS.

    Parameters
    ----------
    G : :class:`networkx.classes.digraph.DiGraph`
        Neuron from swc represented as directed graph.

    node_id : int
        The id of the node to use as a seed.
        If df is not None this become the node index.

    depth : int
        The max depth for BFS to traven in each direction.

    df : None, DataFrame (default = None)
        Dataframe storing indices.
        In some cases indexing by row number is preferred.

    Returns
    -------
    G_sub : :class:`networkx.classes.digraph.DiGraph`
        Subgraph

    tree : DiGraph
        The tree returned by BFS.
    """
    if df is not None:
        node_id = int(df.iloc[node_id]["sample"])
    G_undir = G.to_undirected()
    tree = nx.bfs_tree(G_undir, node_id, depth_limit=depth)  # forward BFS
    G_sub = nx.subgraph(G, list(tree.nodes))
    return G_sub, tree


def swc2skeleton(swc_file, origin=None):
    """Converts swc file into Skeleton object

    Arguments:
        swc_file {str} -- path to SWC file
    Keyword Arguments:
        origin {numpy array with shape (3,1)} -- origin of coordinate frame in microns, (default: None assumes (0,0,0) origin)
    Returns:
        skel {cloudvolume.Skeleton} -- Skeleton object of given SWC file
    """
    with open(swc_file, "r") as f:
        contents = f.read()
    # get every line that starts with a hashtag
    comments = [i.split(" ") for i in contents.split("\n") if i.startswith("#")]
    offset = np.array([float(j) for i in comments for j in i[2:] if "OFFSET" in i])
    color = [float(j) for i in comments for j in i[2].split(",") if "COLOR" in i]
    # set alpha to 0.0 so skeleton  is opaque
    color.append(0.0)
    color = np.array(color, dtype="float32")
    skel = Skeleton.from_swc(contents)
    # physical units
    # space can be 'physical' or 'voxel'
    skel.space = "physical"
    # hard coding parsing the id from the filename
    idx = swc_file.find("G")

    skel.id = int(swc_file[idx + 2 : idx + 5])

    # hard coding changing  data type of vertex_types
    skel.extra_attributes[-1]["data_type"] = "float32"
    skel.extra_attributes.append(
        {"id": "vertex_color", "data_type": "float32", "num_components": 4}
    )
    # add offset to vertices
    # and shift by origin
    # print(skel.vertices[0])
    skel.vertices += offset
    # print(skel.vertices[0])
    if origin is not None:
        skel.vertices -= origin
    # print(skel.vertices[0])
    # convert from microns to nanometers
    skel.vertices *= 1000
    skel.vertex_color = np.zeros((skel.vertices.shape[0], 4), dtype="float32")
    skel.vertex_color[:, :] = color

    return skel
