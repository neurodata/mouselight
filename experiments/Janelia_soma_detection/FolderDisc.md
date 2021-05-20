# Janelia soma detection folder discription
## Purpose:
Record and reproduce the experimental results of ```find_soma()``` on Janelia mouse brain data.
## Contents:
1. ```brain1testResults.xlsx``` Human-monitored examination results on the functionality of the soma detector. The experiments were done at three resolution levels, i.e., mip=1,2, and 3. In addition to providing hit rate, miss rate, and false possitive rate of each experiment, the failure case number, which is termed ```item number``` in the spreadsheet, is also provided.
2. ```CTsoma.py``` operate soma detection on the subvolumes contained in the subfolder ```data```.
3. ```download_voxels_100.py``` generate 174 bounding boxes (bbox) within the Janelia mouse brain data, each of which has a standard subvolume size of 100x100x100 um^3. The bbox information of each subvolume is saved as the .npy file title in the ```data``` folder.
## How to reproduce the result?
- step 1: Use ```download_voxels_100.py``` to generate bbox information. Before running the script, change ```mip``` to a desired value.
- step 2: Use ```CTsoma.py``` to visualize the results. Likewise, before running the script, make sure ```mip``` is set to the desired value.
