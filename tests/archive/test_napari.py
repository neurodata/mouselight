# import pytest
# from brainlit.utils.ngl_pipeline import NeuroglancerSession
# from brainlit.utils.upload import get_volume_info, create_cloud_volume
# from cloudvolume import Bbox
# import numpy as np
# import napari
# import os


# def main():
#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     f = "file://" + str(os.path.join(dir_path, "data_octree"))
#     print(f)

#     _, _, vox_size, tiff_dims = get_volume_info("tests/data_octree", 2, 0)
#     _ = create_cloud_volume(f, tiff_dims, vox_size, 2)
#     sess = NeuroglancerSession(url=f, mip=1)
#     box = Bbox((0, 0, 0), (528, 400, 208))
#     img = sess.pull_bounds_img(box)
#     print(np.sum(img))
#     with napari.gui_qt():
#         sess.napari_viewer(img)


# if __name__ == "__main__":
#     main()
