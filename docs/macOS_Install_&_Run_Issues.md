# Installing brainlit on macOS using conda environment:
1. Installation of conda environment
    * [get Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)
    * create a virtual environment: `conda create --name brainlit python=3.8`
    * activate the environment: `conda activate brainlit`
2. Install brainlit
    * Option 1: install from pypi
        * install brainlit: `pip install brainlit`
    * Option 2: install brainlit from source
        * cd to directory you want to install brainlit in
        * clone the repo: git clone https://github.com/neurodata/brainlit.git
        * cd into the repo: `cd brainlit`
        * install brainlit: `pip install -e .`


# Downloading brains tutorial on macOS using conda environment, using jupyter notebook: 
* in a new terminal: `conda activate brainlit`
* cd to directory brainlit is installed
* `cd brainlit/docs/notebooks/utils`
* run jupyter notebook: `jupyter notebook`
* select `downloading_brains.ipynb`
* run tutorial


## Obstacles Encountered During Brainlit Tutorial (macOS)
(contact jduva4@jhu.edu or akodba1@jhu.edu for related questions)

1. Issues with using a jupyter notebook
    * [fixes](https://jupyter-notebook.readthedocs.io/en/stable/troubleshooting.html)

2. If using ```virtualenv``` to create the environment rather than ```conda```, make sure that you have Python 3 installed outside of Anaconda (call ```python --version```) because many systems will not. Make sure that ```pip``` references Python 3 (the ```pip --version``` command should show ```3.xx``` in the path), otherwise ```pip``` installs could be updating Python 2 exclusively. 

4. May run into a schema-related error when importing napari in Step 1: “This is specifically a suppressible warning because if you’re using a schema other than the ones in SUPPORTED_VERSIONS, it’s possible that not all functionality will be supported properly. If you don’t want to see these messages, add a warningfilter to your code.” (Source: https://github.com/cwacek/python-jsonschema-objects/issues/184)

5. Not exclusive to macOS but make sure aws .json file has no dollar signs in the strings and is being edited/saved within the terminal using a program like Nano or Vim. Do not use external editors like Sublime.

6.  AWS Credendials Issues
    * Refer to the instructions on: https://github.com/NeuroDataDesign/brainlit/blob/develop/docs/AWS_Credentials_Issues.md

7. Section (2) of downloading_brains notebook, Create a Neuroglancer instance and download the volume: make sure variables are correct and functions have correct inputs
    * For Example:
        * Wrong: `img, bbox, vox = ngl_sess.pull_voxel(2, v_id, radius, radius, radius)`
        * Right:  `img, bbox, vox = ngl_sess.pull_voxel(2, v_id, radius)`
    
8. Section (4) of downloading_brains notebook, View the volume: the iPyNb kernel may consistently die when running, not allowing napari to be viewed
    * In terminal, type `pip install opencv-contrib-python-headless`
    * Or try including ```%gui qt``` just above the ```import napari``` line. 