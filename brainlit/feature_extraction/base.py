from abc import abstractmethod
from sklearn.base import BaseEstimator
from brainlit.utils.session import NeuroglancerSession
from brainlit.utils.util import *
import warnings
import numpy as np
import pandas as pd
import time
from cloudvolume import CloudVolume
import feather
from joblib import Parallel, delayed
from typing import Optional, List, Union, Tuple


class BaseFeatures(BaseEstimator):
    """Base class for generating features from precomputed volumes.

    Arguments:
        url: Precompued path either to a file URI or url URI of image data.
        size: A size hyperparameter. For Neighborhoods, this is the radius.
        offset: Added to the coordinates of a positive sample to generate a negative sample.
        segment_url: Precompued path either to a file URI or url URI of segmentation data.

    Attributes:
        url: CloudVolumePrecomputedPath to image data.
        size: A size hyperparameter. In Neighborhoods, this is the radius.
        offset: Added to the coordinates of a positive sample to generate a negative sample.
        download_time: Tracks time taken to download the data.
        conversion_time: Tracks time taken to convert data to features.
        write_time: Tracks time taken to write features to files.
        segment_url: CloudVolumePrecomputedPath to segmentation data.
    """

    def __init__(
        self,
        url: str,
        size: int,
        offset: Tuple[int, int, int] = [15, 15, 15],
        segment_url: Optional[str] = None,
    ):
        check_precomputed(url)
        check_type(size, (int, np.integer))
        if size < 0:
            raise ValueError(f"Size {size} should be nonnegative.")
        check_size(offset)
        if segment_url is not None:
            check_precomputed(segment_url)
        self.url = url
        self.size = size
        self.offset = offset
        self.download_time = 0
        self.conversion_time = 0
        self.write_time = 0
        self.segment_url = segment_url

    @abstractmethod
    def _convert_to_features(self, img: np.ndarray) -> np.ndarray:
        """Computes features from image data.

        Arguments:
            img: Image data.

        Returns:
            features: Feature data.
        """

    def fit(
        self,
        seg_ids: List[int],
        num_verts: Optional[int] = None,
        file_path: Optional[str] = None,
        batch_size: int = 10000,
        start_seg: Optional[int] = None,
        start_vert: int = 0,
        include_neighborhood: bool = False,
        n_jobs: int = 1,
    ) -> np.ndarray:
        """Pulls image and background.

        Arguments:
            seg_ids: A list of segment indices.
            num_verts: If not None, only runs on a set number of vertices. Defaults to None.
            file_path: If not None, then the extracted data will be written directly
                into a feather binary file. The file_path specifies the prefix
                of the file. Defaults to None.
            batch_size: Size of each batch of data to be loaded/written. Only
                used when file_path is not none. Default 10000.
            start_seg: Specifies which segment in the seg_ids list to start at. Default 0.
            start_vert: Specifies which vertex of the first seg_id to start at. Default 0.
            include_neighborhood: If extracting linear features, specifies if the general
                neighborhood should be extracted as well. Defaults to False.
            n_jobs: Number of cores to use. -1 to use all available cores. Defaults to 1.

        Returns:
            df: A dataframe of data containing [segment, vertex, label, f_1, f_2, ..., f_d] columns.
        """
        check_iterable_type(seg_ids, (int, np.integer))
        if num_verts is not None:
            check_type(num_verts, (int, np.integer))
        if file_path is not None:
            check_type(file_path, str)
        check_type(batch_size, (int, np.integer))
        if batch_size < 1:
            raise ValueError(f"Batch size {batch_size} should not be negative.")
        if start_seg is not None:
            check_type(start_seg, (int, np.integer))
            if start_seg < 0:
                raise ValueError(
                    f"Starting segment {start_seg} should not be negative."
                )
        check_type(start_vert, (int, np.integer))
        if start_vert < 0:
            raise ValueError(f"Starting vertex {start_vert} should not be negative.")
        check_type(include_neighborhood, bool)
        check_type(n_jobs, (int, np.integer))
        if n_jobs < 1:
            raise ValueError(f"Number of jobs {n_jobs} should be positive.")

        voxel_dict = {}
        counter = 0
        batch_id = 0
        ngl = NeuroglancerSession(self.url, url_segments=self.segment_url)
        # ngl.cv.progress = False
        ngl.cv_segments.progress = False
        # if self.segment_url is None:
        #     ngl_skel = NeuroglancerSession(self.url)
        # else:
        #     ngl_skel = NeuroglancerSession(self.url, self.segment_url)

        if start_seg is not None:
            seg_ids = seg_ids[seg_ids.index(start_seg) :]

        if file_path is None:
            return self._serial_processing(
                seg_ids, ngl, num_verts, start_vert, include_neighborhood
            )
        else:
            if n_jobs == 1:
                self._serial_processing(
                    seg_ids,
                    ngl,
                    num_verts,
                    start_vert,
                    include_neighborhood,
                    True,
                    batch_size,
                    file_path,
                )
            else:
                start_vertices = [0] * len(seg_ids)
                start_vertices[0] = start_vert
                par = Parallel(n_jobs=n_jobs)
                par(
                    delayed(self._parallel_processing)(
                        seg_id,
                        ngl,
                        ngl_skel,
                        num_verts,
                        start_vertices[i],
                        include_neighborhood,
                        batch_size,
                        file_path,
                    )
                    for i, seg_id in enumerate(seg_ids)
                )

    def _serial_processing(
        self,
        seg_ids,
        ngl,
        num_verts,
        start_vert,
        include_neighborhood,
        write=False,
        batch_size=None,
        file_path=None,
    ):
        """Core code which actually extracts features."""
        voxel_dict = {}
        counter = 0
        batch_id = 0

        for seg_id in seg_ids:
            if self.segment_url is None:
                segment = ngl.cv.skeleton.get(seg_id)
            else:
                cv_skel = CloudVolume(self.segment_url, use_https=True)
                segment = cv_skel.skeleton.get(seg_id)
            if num_verts is not None and num_verts <= len(segment.vertices):
                if num_verts <= len(segment.vertices):
                    verts = segment.vertices[start_vert:num_verts]
                else:
                    warnings.warn(
                        UserWarning(
                            f"Number of vertices {num_verts} greater than total vertices {len(segment.vertices)}. Defaulting to max len."
                        )
                    )
                    verts = segment.vertices[start_vert:]
            else:
                verts = segment.vertices[start_vert:]
            start_vert = 0
            for v_id, vertex in enumerate(verts):

                start = time.time()

                img, bounds, voxel = ngl.pull_voxel(seg_id, v_id, self.size)
                img_off = ngl.pull_bounds_img(bounds + self.offset)

                end = time.time()
                self.download_time += end - start

                start = time.time()

                features = self._convert_to_features(img)
                features_off = self._convert_to_features(img_off)

                end = time.time()
                self.conversion_time += end - start

                voxel_dict[counter] = {
                    **{"Segment": int(seg_id), "Vertex": int(v_id), "Label": 1},
                    **features,
                }
                counter += 1
                voxel_dict[counter] = {
                    **{"Segment": int(seg_id), "Vertex": int(v_id), "Label": 0},
                    **features_off,
                }
                counter += 1
                if write:
                    if counter % batch_size == 0 or (counter + 1) % batch_size == 0:
                        df = pd.DataFrame.from_dict(voxel_dict, "index")
                        path = (
                            file_path
                            + str(batch_id * batch_size)
                            + "_"
                            + str((batch_id + 1) * batch_size)
                            + "_"
                            + str(seg_id)
                            + "_"
                            + str(v_id)
                            + ".feather"
                        )

                        start = time.time()

                        feather.write_dataframe(df, path)

                        end = time.time()
                        self.write_time += end - start

                        voxel_dict = {}
                        batch_id += 1

        if file_path is None:
            if write:
                if not (counter % batch_size == 0 or (counter + 1) % batch_size == 0):
                    df = pd.DataFrame.from_dict(voxel_dict, "index")
                    path = (
                        file_path
                        + str(batch_id * batch_size)
                        + "_"
                        + str(counter)
                        + "_"
                        + str(seg_id)
                        + "_"
                        + str(v_id)
                        + ".feather"
                    )
                    feather.write_dataframe(df, path)
            else:
                df = pd.DataFrame.from_dict(voxel_dict, "index")
            return df

    def _parallel_processing(
        self,
        seg_id,
        ngl,
        num_verts,
        start_vert,
        include_neighborhood,
        batch_size=None,
        file_path=None,
    ):
        voxel_dict = {}
        counter = 0
        batch_id = 0
        if self.segment_url is None:
            segment = ngl.cv.skeleton.get(seg_id)
        else:
            cv_skel = CloudVolume(self.segment_url)
            segment = cv_skel.skeleton.get(seg_id)
        if num_verts is not None:
            verts = segment.vertices[start_vert:num_verts]
        else:
            verts = segment.vertices[start_vert:]
        for v_id, vertex in enumerate(verts):
            img, bounds, voxel = ngl.pull_voxel(seg_id, v_id, self.size)
            img_off = ngl.pull_bounds_img(bounds + self.offset)
            features = self._convert_to_features(img, include_neighborhood)
            features_off = self._convert_to_features(img_off, include_neighborhood)
            voxel_dict[counter] = {
                **{"Segment": int(seg_id), "Vertex": int(v_id), "Label": 1},
                **features,
            }
            counter += 1
            voxel_dict[counter] = {
                **{"Segment": int(seg_id), "Vertex": int(v_id), "Label": 0},
                **features_off,
            }
            counter += 1
            if counter % batch_size == 0 or (counter + 1) % batch_size == 0:
                df = pd.DataFrame.from_dict(voxel_dict, "index")
                path = (
                    file_path
                    + str(batch_id * batch_size)
                    + "_"
                    + str((batch_id + 1) * batch_size)
                    + "_"
                    + str(seg_id)
                    + "_"
                    + str(v_id)
                    + ".feather"
                )

                start = time.time()

                feather.write_dataframe(df, path)

                end = time.time()
                self.write_time += end - start
        return self.download_time, self.conversion_time, self.write_time
