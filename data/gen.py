from pathlib import Path
from brainlit.utils import upload_volumes, upload_segments

# this checks is data exists locally, and generates it if not
# currently deprecated as local data is enforced by being excluded from the .gitignore
top_level = Path(__file__).absolute().parents[0]
input = (top_level / "data_octree").as_posix()
url = (top_level / "test_upload").as_uri()
url_seg = url + "_segments"
url = url + "/serial"
if not (Path(url[5:]) / "info").is_file():
    print("Uploading data.")
    upload_volumes(input, url, 1)
if not (Path(url_seg[5:]) / "info").is_file():
    print("Uploading segmentataion.")
    upload_segments(input, url_seg, 1)
assert (Path(url[5:]) / "info").is_file()
assert (Path(url_seg[5:]) / "info").is_file()
