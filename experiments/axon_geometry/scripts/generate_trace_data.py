import numpy as np
import brainlit
import scipy
import scipy.stats
from brainlit.utils.Neuron_trace import NeuronTrace
from cloudvolume.exceptions import SkeletonDecodeError
from brainlit.algorithms.trace_analysis.fit_spline import GeometricGraph
from brainlit.algorithms.trace_analysis.spline_fxns import curvature, torsion
import os
from pathlib import Path
import pandas as pd
from networkx.readwrite import json_graph
import json
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import networkx as nx
from scipy.cluster.hierarchy import dendrogram


def node_height(G, node):
    predecessors = list(G.predecessors(node))
    L = len(predecessors)
    assert L == 1 or L == 0
    if L == 0:
        return 0
    else:
        return 1 + node_height(G, predecessors[0])


def generate_brain_trace_data(brain: str, spacing: int):
    cwd = Path(os.path.abspath(__file__))
    exp_dir = cwd.parents[1]
    data_dir = os.path.join(exp_dir, "data")
    brain_dir = os.path.join(data_dir, brain)
    segments_swc_dir = os.path.join(brain_dir, "segments_swc")
    trace_data_dir = os.path.join(brain_dir, "trace_data")
    trace_data_dir = os.path.join(trace_data_dir, str(spacing))
    if not os.path.exists(trace_data_dir):
        os.makedirs(trace_data_dir)

    max_id = 300
    print(f"Trace directory: {trace_data_dir}")
    print(f"Spacing: {spacing} microns")
    for i in np.arange(0, max_id):
        i = int(i)
        string_id = str(i).zfill(3)
        seg_swc_path = os.path.join(
            segments_swc_dir,
            "{}_g-{}_consensus.swc".format(
                "2018-08-01" if brain == "brain1" else "2018-12-01", string_id
            ),
        )
        if os.path.exists(seg_swc_path) is True:
            swc_trace = NeuronTrace(path=seg_swc_path)
            df_swc_offset_neuron = swc_trace.get_df()
            print("Loaded segment {}".format(i))
            G = GeometricGraph(df=df_swc_offset_neuron)
            print("Initialized GeometricGraph")
            spline_tree = G.fit_spline_tree_invariant()
            print("Computed splines")

            trace_data_path = os.path.join(trace_data_dir, f"{i}.npy")
            trace_data = np.empty(len(spline_tree.nodes), dtype="object")
            for j, node in enumerate(spline_tree.nodes):
                if j == 0:
                    main_branch = spline_tree.nodes[node]
                spline = spline_tree.nodes[node]
                spline_height = node_height(spline_tree, node)
                if spline_height == 0:
                    spline_class = "primary"
                else:
                    successors = spline_tree.successors(node)
                    if len(list(successors)) == 0:
                        spline_class = "terminal"
                    else:
                        spline_class = "collateral"
                starting_length = spline["starting_length"]
                path = spline["path"]
                tck, u_um = spline["spline"]
                t = tck[0]
                c = tck[1]
                k = tck[2]

                # evaluate segment length (in um)
                seg_length = u_um[-1] - u_um[0]
                # resample points at specified spacing
                uu = np.arange(u_um[0], u_um[-1], spacing)
                # evaluate mean curvature of the segment
                _curvature = curvature(uu, t, c, k)
                mean_curvature = np.mean(_curvature)
                # evaluate mean torsion of the segment
                _torsion = np.abs(torsion(uu, t, c, k))
                mean_torsion = np.mean(_torsion)

                trace_data[j] = {
                    "height": spline_height,
                    "class": spline_class,
                    "seg_length": seg_length,
                    "starting_length": starting_length,
                    "mean_curvature": mean_curvature,
                    "mean_torsion": mean_torsion,
                    "curvature": _curvature,
                    "torsion": _torsion,
                    "path": path,
                }
            np.save(trace_data_path, trace_data)


# spacing of 1um is for autocorrelation plot
# spacing of 14um is for regression plots
for brain in ["brain1", "brain2"]:
    generate_brain_trace_data(brain, 1)
    generate_brain_trace_data(brain, 14)
