import os
import subprocess
from glob import glob
import shlex
import requests as r
import numpy as np
import h5py
from cloudvolume import CloudVolume
from collections import defaultdict
import uuid
from util import aws_cli
import argparse
from scipy.io import loadmat

import pathlib


def loadmat_v73(mat_path):
    arrays = {}
    f = h5py.File(mat_path, "r")
    for k, v in f.items():
        arrays[k] = np.array(v)
    return arrays


class NGLink:
    def __init__(self, json_link):
        self.points = defaultdict(lambda: "")
        self.json_link = json_link
        self._set_json_from_link()

    def get_annotations(self, points):
        annotations = []
        for i, j in points.items():
            x = {
                "point": j.tolist(),
                "type": "point",
                "id": f"{uuid.uuid1().hex}",
                "description": i,
            }
            annotations.append(x)
        return annotations

    def get_points_in(self, coordinate_system):
        if coordinate_system == "voxel":
            return self.points
        else:
            return {i[0]: (i[1] * self.points_voxel_size) for i in self.points.items()}

    def _set_json_from_link(self):
        self._json = r.get(self.json_link).json()
        self._parse_voxel_size()
        self.layers = [self._parse_layer(i) for i in self._json["layers"]]

    def _parse_layer(self, layer_data):
        if layer_data["type"] == "image":
            return self._parse_image_layer(layer_data)
        elif layer_data["type"] == "annotation":
            return self._parse_annotation_layer(layer_data)
        else:
            return

    def _parse_annotation_layer(self, layer_data):
        # points in physical units
        for i in layer_data["annotations"]:
            if i["type"] != "point":
                continue
            if "description" in i.keys():
                self.points[i["description"].strip()] = i["point"]
            else:
                self.points[f"{i['id']}"] = i["point"]
        return layer_data

    def _parse_image_layer(self, layer_data):
        vol = CloudVolume(layer_data["source"]["url"].split("precomputed://")[-1])
        self.image_shape = np.array(vol.scales[0]["size"])
        # converting from nm to um
        self.image_voxel_size = np.array(vol.scales[0]["resolution"]) / 1e3
        self.voxel_origin = self.image_shape / 2
        self.physical_origin = self.voxel_origin * self.image_voxel_size
        return layer_data

    def _parse_voxel_size(self):
        dims = self._json["dimensions"]
        x_size_m, y_size_m, z_size_m = dims["x"][0], dims["y"][0], dims["z"][0]
        # converting from m to um
        self.points_voxel_size = np.array([x_size_m, y_size_m, z_size_m]) * 1e6


class Fiducial:
    def __init__(self, point, orientation, image_shape, voxel_size, description=""):
        """
        point: 3D point in physical space of fiducial (array-like len 3)
        image_size: size in physical units of native res image in each dim (array-like len 3)
        """
        self.image_shape = np.asarray(image_shape)
        self.voxel_size = np.asarray(voxel_size)
        self._set_origin()
        self.point = np.asarray(point) - self.origin
        self.description = description
        self.orientation = orientation

    def _set_origin(self):
        self.origin = (self.image_shape - 1) * self.voxel_size / 2

    def reorient_point(self, out_orient):
        dimension = len(self.point)
        in_orient = str(self.orientation).lower()
        out_orient = str(out_orient).lower()

        inDirection = ""
        outDirection = ""
        orientToDirection = {"r": "r", "l": "r", "s": "s", "i": "s", "a": "a", "p": "a"}
        for i in range(dimension):
            try:
                inDirection += orientToDirection[in_orient[i]]
            except BaseException:
                raise Exception("in_orient '{0}' is invalid.".format(in_orient))

            try:
                outDirection += orientToDirection[out_orient[i]]
            except BaseException:
                raise Exception("out_orient '{0}' is invalid.".format(out_orient))

        if len(set(inDirection)) != dimension:
            raise Exception("in_orient '{0}' is invalid.".format(in_orient))
        if len(set(outDirection)) != dimension:
            raise Exception("out_orient '{0}' is invalid.".format(out_orient))

        order = []
        flip = []
        for i in range(dimension):
            j = inDirection.find(outDirection[i])
            order += [j]
            flip += [in_orient[j] != out_orient[i]]
        new_point = self._flip_point(self.point, axis=flip)
        new_point = new_point[order]
        # update self
        self.point = new_point
        self.orientation = out_orient

        return new_point

    def _reorient_point(self, out_orient):
        dimension = len(self.point)
        in_orient = str(self.orientation).lower()
        out_orient = str(out_orient).lower()

        inDirection = ""
        outDirection = ""
        orientToDirection = {"r": "r", "l": "r", "s": "s", "i": "s", "a": "a", "p": "a"}
        for i in range(dimension):
            try:
                inDirection += orientToDirection[in_orient[i]]
            except BaseException:
                raise Exception("in_orient '{0}' is invalid.".format(in_orient))

            try:
                outDirection += orientToDirection[out_orient[i]]
            except BaseException:
                raise Exception("out_orient '{0}' is invalid.".format(out_orient))

        if len(set(inDirection)) != dimension:
            raise Exception("in_orient '{0}' is invalid.".format(in_orient))
        if len(set(outDirection)) != dimension:
            raise Exception("out_orient '{0}' is invalid.".format(out_orient))

        order = []
        flip = []
        for i in range(dimension):
            j = inDirection.find(outDirection[i])
            order += [j]
            flip += [in_orient[j] != out_orient[i]]
        new_point = self._flip_point(self.point, axis=flip)
        new_point = new_point[order]
        # update self
        self.orientation = out_orient
        self.point = new_point

        return new_point

    def _flip_point(self, point, axis=0):
        tmp_point = point.copy()
        tmp_point[axis] = -point[axis]
        return tmp_point

    def __str__(self):
        return f"{self.description}: [{self.point[0]}, {self.point[1]}, {self.point[2]} ]\norientation: {self.orientation}"


def get_distances(points1, points2):
    distances = {}
    for i in points1.keys():
        try:
            distances[i] = np.linalg.norm(points1[i] - points2[i])
        except KeyError:
            continue
            distances[i] = np.linalg.norm(points1[i] - points2[i.lower()])
    return distances


def compute_regisration_accuracy(
    target_viz_link,
    atlas_viz_link,
    affine_path,
    velocity_path,
    # voxel size of velocity field
    velocity_field_vsize,
    atlas_orientation="PIR",
    target_orientation="LPS",
):
    # get json link from viz link
    target_viz = NGLink(target_viz_link.split("json_url=")[-1])
    atlas_viz = NGLink(atlas_viz_link.split("json_url=")[-1])

    # get origin-centered fiducials from viz link
    atlas_fiducials = [
        Fiducial(
            j,
            atlas_orientation,
            atlas_viz.image_shape,
            atlas_viz.image_voxel_size,
            description=i,
        )
        for i, j in atlas_viz.get_points_in("physical").items()
    ]
    target_fiducials = [
        Fiducial(
            j,
            target_orientation,
            target_viz.image_shape,
            target_viz.image_voxel_size,
            description=i,
        )
        for i, j in target_viz.get_points_in("physical").items()
    ]

    # run matlab command to get transformed fiducials
    if affine_path != "" and velocity_path != "":
        points = [i.point for i in target_fiducials]
        points_string = [", ".join(map(str, i)) for i in points]
        points_string = "; ".join(points_string)
        # velocity field voxel size
        v_size = ", ".join(str(i) for i in velocity_field_vsize)
        # get current file path and set path to transform_points
        # base_path = pathlib.Path(__file__).parent.parent.absolute() / 'registration'
        base_path = os.path.expanduser("~/CloudReg/registration")
        transformed_points_path = "./transformed_points.mat"

        matlab_command = f"""
            matlab -nodisplay -nosplash -nodesktop -r \"addpath(\'{base_path}\');Aname=\'{affine_path}\';vname=\'{velocity_path}\';v_size=[{v_size}];points=[{points_string}];points_t = transform_points(points,Aname,vname,v_size,\'atlas\');save(\'./transformed_points.mat\',\'points_t\');exit;\"
        """
        print(matlab_command)
        subprocess.run(
            shlex.split(matlab_command),
        )

        # transformed_points.m created now
        points_t = loadmat(transformed_points_path)["points_t"]
        points = {i.description: j for i, j in zip(target_fiducials, points_t)}
    else:
        points = {i.description: i.point for i in target_fiducials}

    atlas_points = {i.description: i.point for i in atlas_fiducials}
    distances = get_distances(atlas_points, points)
    [print(i, j) for i, j in distances.items()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Compute registration accuracy given 2 sets of fiducials from target to atlas"
    )
    parser.add_argument(
        "-target_viz_link", help="viz link to target with fiducials labelled.", type=str
    )
    parser.add_argument(
        "-atlas_viz_link", help="viz link to atlas with fiducials labelled", type=str
    )
    parser.add_argument(
        "--affine_path",
        help="S3 path or local path to matlab transformation files. These will be downloaded to compute the fiducial accuracy",
        type=str,
        default="",
    )
    parser.add_argument(
        "--velocity_path",
        help="S3 path ot local matlab transformation files. These will be downloaded to compute the fiducial accuracy",
        type=str,
        default="",
    )
    parser.add_argument(
        "--velocity_voxel_size",
        help="Voxel size of velocity field in microns",
        nargs="+",
        type=float,
        default=[50.0] * 3,
    )
    parser.add_argument(
        "--atlas_orientation",
        help="3-letter orientation of the atlas data. Default is PIR for Allen Reference Atlas.",
        type=str,
        default="PIR",
    )
    parser.add_argument(
        "--target_orientation",
        help="3-letter orientation of the target data. Default is LPS.",
        type=str,
        default="LPS",
    )
    # parser.add_argument('-ssh_key_path', help='path to identity file used to ssh into given instance')
    # parser.add_argument('-instance_id', help='EC2 Instance ID of instance to run COLM pipeline on.')
    # parser.add_argument('--instance_type', help='EC2 instance type to run pipeline on. minimum r5d.16xlarge',  type=str, default='r5d.16xlarge')

    args = parser.parse_args()

    if args.affine_path.startswith("s3://"):
        # download affine mat to local storage
        aws_cli(shlex.split(f"s3 cp {args.affine_path} ./A.mat"))
        args.affine_path = "./A.mat"
    if args.velocity_path.startswith("s3://"):
        # download velocity mat to local storage
        aws_cli(shlex.split(f"s3 cp {args.velocity_path} ./v.mat"))
        args.velocity_path = "./v.mat"

    compute_regisration_accuracy(
        args.target_viz_link,
        args.atlas_viz_link,
        args.affine_path,
        args.velocity_path,
        args.velocity_voxel_size,
        args.atlas_orientation,
        args.target_orientation,
    )
