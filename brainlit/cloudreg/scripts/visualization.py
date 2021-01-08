import json
import requests
from cloudvolume import CloudVolume
import numpy as np
from util import S3Url
import copy

# 100 um
output_dimensions = [1e-4, 1e-4, 1e-4]

minimum_ngl_json = {
    "dimensions": {
        "x": [output_dimensions[0], "m"],
        "y": [output_dimensions[1], "m"],
        "z": [output_dimensions[2], "m"],
    },
    "layers": [
        {
            "type": "image",
            "source": {
                "url": "",
                "transform": {
                    # last column here is x, y, z translations respectively
                    "matrix": [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                    ],
                    "outputDimensions": {
                        "x": [output_dimensions[0], "m"],
                        "y": [output_dimensions[1], "m"],
                        "z": [output_dimensions[2], "m"],
                    },
                },
            },
            "tab": "source",
            "shader": '#uicontrol vec3 color color(default="white")\n#uicontrol float min slider(default=0, min=0, max=1, step=0.001)\n#uicontrol float max slider(default=1, min=0, max=1, step=0.001)\n#uicontrol float brightness slider(default=0, min=-1, max=1, step=0.1)\n#uicontrol float contrast slider(default=0, min=-3, max=3, step=0.1)\n\nfloat scale(float x) {\n  return (x - min) / (max - min);\n}\n\nvoid main() {\n  emitRGB(\n    color * vec3(\n      scale(\n        toNormalized(getDataValue()))\n       + brightness) * exp(contrast)\n  );\n}',
            "shaderControls": {"max": 0.050},
            "blend": "default",
            "name": "channel",
        },
    ],
    "gpuMemoryLimit": 2000000000,
    "jsonStateServer": "https://json.neurodata.io/v1",
    "layout": "xy",
}

# desired ara resolution in microns
ara_average_data_link = (
    lambda res: f"https://open-neurodata.s3.amazonaws.com/ara_2016/sagittal_{res}um/average_{res}um"
)
ara_annotation_data_link = (
    lambda res: f"https://open-neurodata.s3.amazonaws.com/ara_2016/sagittal_{res}um/annotation_{res}um_2017"
)


def create_viz_link(
    s3_layer_paths,
    affine_matrices=None,
    shader_controls=None,
    url="https://json.neurodata.io/v1",
    neuroglancer_link="https://ara.viz.neurodata.io/?json_url=",
    output_resolution=np.array([1e-4] * 3),
):
    json_data = get_neuroglancer_json(
        s3_layer_paths, affine_matrices, output_resolution
    )
    r = requests.post(url, json=json_data)
    json_url = r.json()["uri"]
    viz_link = f"{neuroglancer_link}{json_url}"
    return viz_link


def get_neuroglancer_json(s3_layer_paths, affine_matrices, output_resolution):
    """
    s3_layer_paths must be a list of s3 paths to precomputed volumes you want to visualize
    affine_matrices must be a list of 4x4 np.arrays that contain the transformation for each layer and is the same length as s3_layer_paths
    """
    ngl_json = copy.deepcopy(minimum_ngl_json)
    ngl_json["layers"] = []
    if affine_matrices is None:
        affine_matrices = [None] * len(s3_layer_paths)
    ngl_json["layers"] = [
        get_layer_json(i, j, output_resolution)
        for i, j in zip(s3_layer_paths, affine_matrices)
    ]
    # print(ngl_json['layers'][0]['source'])
    # print(ngl_json['layers'][1]['source'])
    return ngl_json


def get_output_dimensions_json(output_resolution):
    # output dimensions must be  in meters
    output_json = copy.deepcopy(minimum_ngl_json["dimensions"])
    output_json["x"][0] = output_resolution[0]
    output_json["y"][0] = output_resolution[1]
    output_json["z"][0] = output_resolution[2]
    return output_json


def get_layer_json(s3_layer_path, affine_matrix, output_resolution):
    """
    affine_matrix has translations in microns
    output resolution in meters
    """
    vol = CloudVolume(s3_layer_path)
    s3_url = S3Url(s3_layer_path)
    # this is in units of m
    # output_resolution = np.array([minimum_ngl_json['dimensions']['x'][0], minimum_ngl_json['dimensions']['y'][0], minimum_ngl_json['dimensions']['z'][0]])

    if affine_matrix is None:
        affine_matrix = np.eye(4)
    else:
        # convert translations from microns to voxels and convert output resolution from m to um
        affine_matrix[:3, -1] /= output_resolution * 1e6

    if s3_url.bucket == "colm-precomputed-volumes":
        url = f"precomputed://https://dlab-colm.neurodata.io/{s3_url.key}"
    else:
        url = f"precomputed://{s3_layer_path}"

    # layer_data['source']['transform']['matrix'] = affine[:3,:].tolist()
    layer_data = {
        "type": vol.layer_type,
        "source": {
            "url": url,
            "transform": {
                # last column here is x, y, z translations respectively
                "matrix": affine_matrix[:3, :].tolist(),
                "outputDimensions": get_output_dimensions_json(output_resolution),
            },
        },
        "tab": "source",
        "shader": '#uicontrol vec3 color color(default="white")\n#uicontrol float min slider(default=0, min=0, max=1, step=0.001)\n#uicontrol float max slider(default=1, min=0, max=1, step=0.001)\n#uicontrol float brightness slider(default=0, min=-1, max=1, step=0.1)\n#uicontrol float contrast slider(default=0, min=-3, max=3, step=0.1)\n\nfloat scale(float x) {\n  return (x - min) / (max - min);\n}\n\nvoid main() {\n  emitRGB(\n    color * vec3(\n      scale(\n        toNormalized(getDataValue()))\n       + brightness) * exp(contrast)\n  );\n}',
        "shaderControls": {"max": 0.005},
        "blend": "default",
        "name": s3_url.key.split("/")[-1],
    }

    return layer_data
