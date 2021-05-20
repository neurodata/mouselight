import numpy as np
import re
import pandas as pd
import networkx as nx
from cloudvolume import CloudVolume, Skeleton
from io import StringIO
import os
from brainlit.utils.util import (
    check_type,
    check_size,
)
from sklearn.metrics import pairwise_distances_argmin_min
import warnings


class NeuronTrace:
    """Neuron Trace class to handle neuron traces as swcs and s3 skeletons

    Arguments
    ---------
        path : str
            Path to either s3 bucket (url) or swc file (filepath).
        seg_id : int
            If s3 bucket path is provided, the segment number to pull, default None.
        mip : int
            If s3 bucket path is provided, the resolution to use for scaling, default None.
        rounding : bool
            If s3 is provided, specifies if it should be rounded, default True
        read_offset : bool
            If swc is provided, whether offset should be read from file, default False.
        fill_missing: bool
            Always passes directly into 'CloudVolume()' function to fill missing skeleton values with 0s, default True.
        use_https : bool
            Always passes directly into 'CloudVolume()' function to set use_https to desired value, default True.

    Attributes
    ----------
        path : str
            Path to either s3 bucket (url) or swc file (filepath)
        input_type : bool
            Specifies whether input file is 'swc' or 'skel'
        df : :class:`pandas.DataFrame`
            Indices, coordinates, and parents of each node
        args : tuple
            Stores arguments for df - offset, color, cc, branch
        seg_id : int
            If s3 bucket path is provided, the segment number to pull
        mip : None,int
            If s3 bucket path is provided, the resolution to use for scaling

    Example
    ----------
    >>> swc_path = "./data/data_octree/consensus-swcs/2018-08-01_G-002_consensus.swc"
    >>> s3_path = "s3://open-neurodata/brainlit/brain1_segments"
    >>> seg_id = 11
    >>> mip = 2

    >>> swc_trace = NeuronTrace(swc_path)
    >>> s3_trace = NeuronTrace(s3_path,seg_id,mip)

    """

    def __init__(
        self,
        path,
        seg_id=None,
        mip=None,
        rounding=True,
        read_offset=False,
        fill_missing=True,
        use_https=False,
    ):
        self.path = path
        self.input_type = None
        self.df = None
        self.args = []
        self.seg_id = seg_id
        self.mip = mip
        self.rounding = rounding
        self.fill_missing = fill_missing
        self.use_https = use_https

        check_type(path, str)
        check_type(seg_id, (type(None), int))
        check_type(mip, (type(None), int))
        check_type(read_offset, bool)
        check_type(rounding, bool)
        if (seg_id == None and type(mip) == int) or (
            type(seg_id) == int and mip == None
        ):
            raise ValueError(
                "For 'swc' do not input mip or seg_id, and for 'skel', provide both mip and seg_id"
            )

        # first check if it is a skel
        if seg_id != None and mip != None:
            cv = CloudVolume(
                path, mip=mip, fill_missing=fill_missing, use_https=use_https
            )
            skeleton = cv.skeleton.get(seg_id)
            if type(skeleton) is Skeleton:
                self.input_type = "skel"

        # else, check if it is a swc by checking if file exists/extension is .swc
        elif os.path.isfile(self.path) and os.path.splitext(path)[-1].lower() == ".swc":
            self.input_type = "swc"

        # if it is not a swc or skeleton, raise error
        if self.input_type != "swc" and self.input_type != "skel":
            raise ValueError("Did not input 'swc' filepath or 'skel' url")

        # next, convert to a dataframe
        if self.input_type == "swc" and read_offset == False:
            df, offset, color, cc, branch = self._read_swc(self.path)
            args = [offset, color, cc, branch]
            self.df = df
            self.args = args

        elif self.input_type == "swc" and read_offset == True:
            df, color, cc, branch = self._read_swc_offset(path)
            args = [None, color, cc, branch]
            self.df = df
            self.args = args

        elif self.input_type == "skel":
            df = self._read_s3(path, seg_id, mip, rounding)
            (self.path, seg_id, mip)
            self.df = df

    # public methods
    def get_df_arguments(self):
        """Gets arguments for df - offset, color, cc, branch

        Returns
        -------
            self.args : list
                list of arguments for df, if found - offset, color, cc, branch

        Example
        -------
        >>> swc_trace.get_df_arguments()
        >>> [[73954.8686, 17489.532566, 34340.365689], [1.0, 1.0, 1.0], nan, nan]
        """
        return self.args

    def get_df(self):
        """Gets the dataframe providing indices, coordinates, and parents of each node

        Returns
        -------
            self.df : :class:`pandas.DataFrame`
                dataframe providing indices, coordinates, and parents of each node

        Example
        -------
        >>> swc_trace.get_df()
        >>> sample    structure    x    y    z    r    parent
            0    1    0    -52.589700    -1.448032    -1.228827    1.0    -1
            1    2    0    -52.290940    -1.448032    -1.228827    1.0    1
            2    3    0    -51.992181    -1.143616    -0.240423    1.0    2
            3    4    0    -51.095903    -1.143616    -0.240423    1.0    3
            4    5    0    -50.797144    -0.839201    -0.240423    1.0    4
            ...    ...    ...    ...    ...    ...    ...    ...
            148    149    0    45.702088    14.381594    -7.159252    1.0    148
            149    150    0    46.000847    14.686010    -7.159252    1.0    149
            150    151    0    46.897125    14.686010    -7.159252    1.0    150
            151    152    0    47.494643    15.294842    -7.159252    1.0    151
            152    153    6    48.092162    15.294842    -7.159252    1.0    152
            53 rows × 7 columns
        """
        return self.df

    def get_skel(self, benchmarking=False, origin=None):
        """Gets a skeleton version of dataframe, if swc input is provided

        Arguments
        ----------
            origin : None, numpy array with shape (3,1) (default = None)
                origin of coordinate frame in microns, (default: None assumes (0,0,0) origin)
            benchmarking : bool
                For swc files, specifies whether swc file is from benchmarking dataset, to obtain skeleton ID
        Returns
        --------
            skel : cloudvolume.Skeleton
                Skeleton object of given SWC file

        Example
        -------
        >>> swc_trace.get_skel(benchmarking=True)
        >>> Skeleton(segid=, vertices=(shape=153, float32), edges=(shape=152, uint32), radius=(153, float32), vertex_types=(153, uint8), vertex_color=(153, float32), space='physical' transform=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        """
        check_type(origin, (type(None), np.ndarray))
        check_type(benchmarking, bool)
        if type(origin) == np.ndarray:
            check_size(origin)

        if self.input_type == "swc":
            skel = self._swc2skeleton(self.path, benchmarking, origin)
            return skel
        elif self.input_type == "skel":
            cv = CloudVolume(
                self.path,
                mip=self.mip,
                fill_missing=self.fill_missing,
                use_https=self.use_https,
            )
            skel = cv.skeleton.get(self.seg_id)
            return skel

    def get_df_voxel(self, spacing, origin=np.array([0, 0, 0])):
        """Converts coordinates in pd.DataFrame from spatial units to voxel units

        Arguments
        ----------
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

        Example
        -------
        >>> swc_trace.get_df_voxel(spacing=np.asarray([2,2,2]))
        >>> sample    structure    x    y    z    r    parent
            0    1    0    -26    -1    -1    1.0    -1
            1    2    0    -26    -1    -1    1.0    1
            2    3    0    -26    -1    0    1.0    2
            3    4    0    -26    -1    0    1.0    3
            4    5    0    -25    0    0    1.0    4
            ...    ...    ...    ...    ...    ...    ...    ...
            148    149    0    23    7    -4    1.0    148
            149    150    0    23    7    -4    1.0    149
            150    151    0    23    7    -4    1.0    150
            151    152    0    24    8    -4    1.0    151
            152    153    6    24    8    -4    1.0    152
            153 rows × 7 columns


        """
        check_type(spacing, np.ndarray)
        check_size(spacing)
        check_type(origin, np.ndarray)
        check_size(origin)

        df_voxel = self._df_in_voxel(self.df, spacing, origin)
        return df_voxel

    def get_graph(self, spacing=None, origin=None):
        """Converts dataframe in either spatial or voxel coordinates into a directed graph.
        Will convert to voxel coordinates if spacing is specified.

        Arguments
        ----------
        spacing : None, :class:`numpy.array` (default = None)
            Conversion factor (spatial units/voxel). Assumed to be np.array([x,y,z]).
            Provided if graph should convert to voxel coordinates first. Default is None.
        origin : None, :class:`numpy.array` (default = None)
            Origin of the spatial coordinate, if converting to voxels. Default is None.
            Assumed to be np.array([x,y,z])

        Returns
        -------
        G : :class:`networkx.classes.digraph.DiGraph`
            Neuron from swc represented as directed graph. Coordinates x,y,z are
            node attributes accessed by keys 'x','y','z' respectively.

        Example
        -------
        >>> swc_trace.get_graph()
        >>> <networkx.classes.digraph.DiGraph at 0x7f81a83937f0>
        """
        check_type(spacing, (type(None), np.ndarray))
        if type(spacing) == np.ndarray:
            check_size(spacing)
        check_type(origin, (type(None), np.ndarray))
        if type(origin) == np.ndarray:
            check_size(origin)

        # if origin isn't specified but spacing is, set origin to np.array([0, 0, 0])
        if type(spacing) == np.ndarray and origin is None:
            origin = np.array([0, 0, 0])

        # voxel conversion option
        if type(spacing) == np.ndarray:
            df_voxel = self._df_in_voxel(self.df, spacing, origin)
            G = self._df_to_graph(df_voxel)

        # no voxel conversion option
        else:
            G = self._df_to_graph(self.df)
        return G

    def get_paths(self, spacing=None, origin=None):
        """Converts dataframe in either spatial or voxel coordinates into a list of paths.
        Will convert to voxel coordinates if spacing is specified.

        Arguments
        ----------
        spacing : None, :class:`numpy.array` (default = None)
            Conversion factor (spatial units/voxel). Assumed to be np.array([x,y,z]).
            Provided if graph should convert to voxel coordinates first.  Default is None.
        origin : None, :class:`numpy.array`
            Origin of the spatial coordinate, if converting to voxels. Default is None.
            Assumed to be np.array([x,y,z])

        Returns
        -------
        paths : list
            List of Nx3 numpy.array. Rows of the array are 3D coordinates in voxel
            units. Each array is one path.

        Example
        -------
        >>> swc_trace.get_paths()[0][1:10]
        >>> array([[-52, -1, -1],
                    [-51, -1, 0],
                    [-51, -1, 0],
                    [-50, 0, 0],
                    [-50, 0, 0],
                    [-49, 0, 0],
                    [-48, 0, 0],
                    [-46, 0, 0],
                    [-46, 0, 0]], dtype=object)
        """
        check_type(spacing, (type(None), np.ndarray))
        if type(spacing) == np.ndarray:
            check_size(spacing)
        check_type(origin, (type(None), np.ndarray))
        if type(origin) == np.ndarray:
            check_size(origin)

        # if origin isn't specified but spacing is, set origin to np.array([0, 0, 0])
        if type(spacing) == np.ndarray and origin is None:
            origin = np.array([0, 0, 0])

        # voxel conversion option
        if type(spacing) == np.ndarray:
            df_voxel = self._df_in_voxel(self.df, spacing, origin)
            G = self._df_to_graph(df_voxel)

        # no voxel conversion option
        else:
            G = self._df_to_graph(self.df)

        paths = self._graph_to_paths(G)

        return paths

    def generate_df_subset(
        self, vox_in_img_list, subneuron_start=None, subneuron_end=None
    ):
        """Read a new subset dataframe in coordinates in img spacing.
        Specify specific range of vertices from dataframe if desired

        Arguments
        ----------
        vox_in_img_list : list
            List of voxels
        subneuron_start : None, int (default = None)
            Provides start index, if specified, to apply function to a portion of the dataframe
            Default is None.
        subneuron_end : None, int (default = None)
            Provides end index, if specified, to apply function to a portion of the dataframe
            Default is None.

        Returns
        -------
        df : :class:`pandas.DataFrame`
            Indicies, coordinates (in img spacing) and parents of each node.
            Coordinates are in spatial units.

        Example
        -------
        >>> #swc input, subneuron_start and subneuron_end specified

        >>> subneuron_start = 5
        >>> subneuron_end = 8

        >>> #generate vox_in_img_list
        >>> my_list = []
        >>>for i in range(subneuron_end-subneuron_start):
            my_list.append(10)
        >>> vox_in_img_list_2 = list([my_list,my_list,my_list])

        >>>swc_trace.generate_df_subset(vox_in_img_list_2,subneuron_start,subneuron_end)

        >>> sample    structure    x    y    z    r    parent
                5    6    0    10    10    10    1.0    5
                6    7    0    10    10    10    1.0    6
                7    8    0    10    10    10    1.0    7
        """
        check_type(vox_in_img_list, list)
        check_type(subneuron_start, (type(None), int))
        check_type(subneuron_end, (type(None), int))

        if (subneuron_start == None and type(subneuron_end) == int) or (
            type(subneuron_start) == int and subneuron_end == None
        ):
            raise ValueError(
                "Provide both starting and ending vertices to use for the subneuron"
            )

        # no subneuron range specified
        df = self.df

        # subneuron range specified
        if subneuron_start != None and subneuron_end != None:
            subneuron_df = self.df[subneuron_start:subneuron_end]
            df = subneuron_df

        df_new = self._generate_df_subset(df, vox_in_img_list)

        return df_new

    def get_bfs_subgraph(self, node_id, depth, df=None, spacing=None, origin=None):
        """
         Creates a spanning subgraph from a seed node and parent graph using BFS.

        Arguments
         ----------
         node_id : int
             The id of the node to use as a seed.
             If df is not None this become the node index.
         depth : int
             The max depth for BFS to traven in each direction.
         df : None, DataFrame (default = None)
             Dataframe storing indices.
             In some cases indexing by row number is preferred.
         spacing : None, :class:`numpy.array` (default = None)
             Conversion factor (spatial units/voxel). Assumed to be np.array([x,y,z]).
             Provided if graph should convert to voxel coordinates first.  Default is None.
         origin : :class:`numpy.array`
             Origin of the spatial coordinate, if converting to voxels. Default is None.
             Assumed to be np.array([x,y,z])

         Returns
         -------
         G_sub : :class:`networkx.classes.digraph.DiGraph`
             Subgraph

         tree : DiGraph
             The tree returned by BFS.

         paths : list
            List of Nx3 numpy.array. Rows of the array are 3D coordinates in voxel
            units. Each array is one path.

        Example
        -------
        >>> #swc input, specify node_id and depth
        >>> swc_trace.get_bfs_subgraph(node_id=11,depth=2)
        >>>(<networkx.classes.digraph.DiGraph at 0x7f7f2ce65670>,
            <networkx.classes.digraph.DiGraph at 0x7f7f2ce65370>,
            array([array([[4727, 4440, 3849],
                        [4732, 4442, 3850],
                        [4739, 4455, 3849]]),
                        array([[4732, 4442, 3850],
                        [4749, 4439, 3856]])], dtype=object))
        """

        check_type(node_id, (list, int))
        check_type(depth, int)
        check_type(df, (type(None), pd.core.frame.DataFrame))

        check_type(spacing, (type(None), np.ndarray))
        if type(spacing) == np.ndarray:
            check_size(spacing)
        check_type(origin, (type(None), np.ndarray))
        if type(origin) == np.ndarray:
            check_size(origin)

        # if origin isn't specified but spacing is, set origin to np.array([0, 0, 0])
        if type(spacing) == np.ndarray and origin is None:
            origin = np.array([0, 0, 0])

        # voxel conversion option
        if type(spacing) == np.ndarray:
            df_voxel = self._df_in_voxel(self.df, spacing, origin)
            G = self._df_to_graph(df_voxel)

        # no voxel conversion option
        else:
            G = self._df_to_graph(self.df)

        G_sub, tree = self._get_bfs_subgraph(G, node_id, depth, df)

        paths = self._graph_to_paths(G_sub)

        return G_sub, tree, paths

    def get_sub_neuron(self, bounding_box, spacing=None, origin=None):
        """Returns sub-neuron with node coordinates bounded by start and end

        Arguments
        ----------
        bounding_box : tuple or list or None
            Defines a bounding box around a sub-region around the neuron. Length 2
            tuple/list. First element is the coordinate of one corner (inclusive)
            and second element is the coordinate of the opposite corner (exclusive).
            Both coordinates are numpy.array([x,y,z])in voxel units.
        spacing : None, :class:`numpy.array` (default = None)
            Conversion factor (spatial units/voxel). Assumed to be np.array([x,y,z]).
            Provided if graph should convert to voxel coordinates first.  Default is None.
        origin : :class:`numpy.array`
            Origin of the spatial coordinate, if converting to voxels. Default is None.
            Assumed to be np.array([x,y,z])
        Returns
        -------
        G_sub : :class:`networkx.classes.digraph.DiGraph`
            Neuron from swc represented as directed graph. Coordinates x,y,z are
            node attributes accessed by keys 'x','y','z' respectively.

        Example
        -------
        >>> bounding_box=[[1,2,4],[1,2,3]]

        >>> #swc input, no spacing and origin
        >>> swc_trace.get_sub_neuron(bounding_box)
        >>> <networkx.classes.digraph.DiGraph at 0x7f81a95d1e50>
        """

        check_type(bounding_box, (tuple, list))

        if len(bounding_box) != 2:
            raise ValueError("Bounding box must be length 2")
        check_type(spacing, (type(None), np.ndarray))

        check_type(spacing, (type(None), np.ndarray))
        if type(spacing) == np.ndarray:
            check_size(spacing)
        check_type(origin, (type(None), np.ndarray))
        if type(origin) == np.ndarray:
            check_size(origin)

        # if origin isn't specified but spacing is, set origin to np.array([0, 0, 0])
        if type(spacing) == np.ndarray and origin is None:
            origin = np.array([0, 0, 0])

        # voxel conversion option
        if type(spacing) == np.ndarray:
            df_voxel = self._df_in_voxel(self.df, spacing, origin)
            G = self._df_to_graph(df_voxel)

        # no voxel conversion option
        else:
            G = self._df_to_graph(self.df)

        G_sub = self._get_sub_neuron(G, bounding_box)

        return G_sub

    def get_sub_neuron_paths(self, bounding_box, spacing=None, origin=None):
        """Returns sub-neuron with node coordinates bounded by start and end

        Arguments
        ----------
        bounding_box : tuple or list or None
            Defines a bounding box around a sub-region around the neuron. Length 2
            tuple/list. First element is the coordinate of one corner (inclusive)
            and second element is the coordinate of the opposite corner (exclusive).
            Both coordinates are numpy.array([x,y,z])in voxel units.
        spacing : None, :class:`numpy.array` (default = None)
            Conversion factor (spatial units/voxel). Assumed to be np.array([x,y,z]).
            Provided if graph should convert to voxel coordinates first.  Default is None.
        origin : :class:`numpy.array`
            Origin of the spatial coordinate, if converting to voxels. Default is None.
            Assumed to be np.array([x,y,z])
        Returns
        -------
        paths : list
            List of Nx3 numpy.array. Rows of the array are 3D coordinates in voxel
            units. Each array is one path.

        Example
        -------
        >>> bounding_box=[[1,2,4],[1,2,3]]

        >>> #swc input, no spacing and origin
        >>> swc_trace.get_sub_neuron_paths(bounding_box)
        >>> array([], dtype=object)

        """

        check_type(bounding_box, (tuple, list))

        if len(bounding_box) != 2:
            raise ValueError("Bounding box must be length 2")
        check_type(spacing, (type(None), np.ndarray))

        check_type(spacing, (type(None), np.ndarray))
        if type(spacing) == np.ndarray:
            check_size(spacing)
        check_type(origin, (type(None), np.ndarray))
        if type(origin) == np.ndarray:
            check_size(origin)

        # if origin isn't specified but spacing is, set origin to np.array([0, 0, 0])
        if type(spacing) == np.ndarray and origin is None:
            origin = np.array([0, 0, 0])

        # voxel conversion option
        if type(spacing) == np.ndarray:
            df_voxel = self._df_in_voxel(self.df, spacing, origin)
            G = self._df_to_graph(df_voxel)

        # no voxel conversion option
        else:
            G = self._df_to_graph(self.df)

        G_sub = self._get_sub_neuron(G, bounding_box)

        paths = self._graph_to_paths(G_sub)

        return paths

    @staticmethod
    def ssd(pts1, pts2):
        """Compute significant spatial distance metric between two traces as defined in APP1.
        Args:
            pts1 (np.array): array containing coordinates of points of trace 1. shape: npoints x ndims
            pts2 (np.array): array containing coordinates of points of trace 1. shape: npoints x ndims
        Returns:
            [float]: significant spatial distance as defined by APP1

        Example
        -------
        >>> pts1 = swc_trace.get_paths()[0][1:10]
        >> pts2 = swc_trace.get_paths()[0][11:20]

        >>> NeuronTrace.ssd(pts1,pts2)

        >>>6.247937554557103

        """
        check_type(pts1, np.ndarray)
        check_type(pts2, np.ndarray)

        _, dists1 = pairwise_distances_argmin_min(pts1, pts2)
        dists1 = dists1[dists1 >= 2]
        _, dists2 = pairwise_distances_argmin_min(pts2, pts1)
        dists2 = dists2[dists2 >= 2]
        # If there are is no significant distance between the 2 sets
        if len(dists1) == 0 and len(dists2) == 0:
            ssd = 0
        # Else, calculate the mean
        else:
            dists = np.concatenate([dists1, dists2])
            ssd = np.mean(dists)

        return ssd

    # private methods
    def _read_swc(self, path):
        """
        Read a single swc file

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
            warnings.warn("No offset information found in: " + path)
            offset = [float(0) for i in range(3)]
        # read coordinates
        df = pd.read_table(
            path,
            names=["sample", "structure", "x", "y", "z", "r", "parent"],
            skiprows=header_length,
            delimiter="\s+",
        )
        return df, offset, color, cc, branch

    def _read_swc_offset(self, path):
        df, offset, color, cc, branch = self._read_swc(path)
        df["x"] = df["x"] + offset[0]
        df["y"] = df["y"] + offset[1]
        df["z"] = df["z"] + offset[2]

        return df, color, cc, branch

    def _read_s3(self, s3_path, seg_id, mip, rounding=True):
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
        rounding: bool, Optional
            True is default, false if swc shouldn't be rounded

        Returns
        -------
        df : :class:`pandas.DataFrame`
            Indicies, coordinates, and parents of each node in the swc.
            Coordinates are in spatial units.
        """
        # TODO check header length

        # check input
        cv = CloudVolume(
            s3_path, mip=mip, fill_missing=self.fill_missing, use_https=self.use_https
        )
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

        # round swc files when reading
        if rounding == True:
            res = cv.scales[mip]["resolution"]
            df["x"] = np.round(df["x"] / res[0])
            df["y"] = np.round(df["y"] / res[1])
            df["z"] = np.round(df["z"] / res[2])

        return df

    def _generate_df_subset(self, swc_df, vox_in_img_list):
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
            vox_in_img_list[:][0],
            vox_in_img_list[:][1],
            vox_in_img_list[:][2],
        )

        return df_new

    def _space_to_voxel(self, spatial_coord, spacing, origin=np.array([0, 0, 0])):
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

    def _df_in_voxel(self, df, spacing, origin=np.array([0, 0, 0])):
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
            vox = self._space_to_voxel(row[["x", "y", "z"]].to_numpy(), spacing, origin)
            x.append(vox[0])
            y.append(vox[1])
            z.append(vox[2])

        df_voxel["x"] = x
        df_voxel["y"] = y
        df_voxel["z"] = z

        return df_voxel

    def _df_to_graph(self, df_voxel):
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

    def _get_sub_neuron(self, G, bounding_box):
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

    def _graph_to_paths(self, G):
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
            longest = nx.algorithms.dag.dag_longest_path(
                G_cp
            )  # list of nodes on the path
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

    def _get_bfs_subgraph(self, G, node_id, depth, df=None):
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

    def _swc2skeleton(self, swc_file, benchmarking=False, origin=None):
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

        if benchmarking == True:
            idx1 = swc_file.find(
                "_", swc_file.find("_") + 1
            )  # finding second occurence of "_"
            idx2 = swc_file.find(".")
            skel.id = swc_file[idx1 + 1 : idx2]
        else:
            skel.id = int(swc_file[idx + 2 : idx + 5])

        # hard coding changing  data type of vertex_types
        skel.extra_attributes[-1]["data_type"] = "float32"
        skel.extra_attributes.append(
            {"id": "vertex_color", "data_type": "float32", "num_components": 4}
        )
        # add offset to vertices
        # and shift by origin
        skel.vertices += offset
        if origin is not None:
            skel.vertices -= origin
        # convert from microns to nanometers
        skel.vertices *= 1000
        skel.vertex_color = np.zeros((skel.vertices.shape[0], 4), dtype="float32")
        skel.vertex_color[:, :] = color

        return skel
