
Deep-Learning-Based Neuron Detection in Whole-Brain Volumes
--------------------------------

The three scripts in this directory test Prerna Singh's 3D-CNN on three different datasets from Brain 1:
  - ``compareSubvols.ipynb``: The middle-most volume within the brain (1020x1020x1020 voxels)
  - ``testJacoData.ipynb``  : 174 subvolumes (from Jaco's experiment) that contain all 180 known cells
  - ``wholeBrain.ipynb``    : The entire Brain 1 dataset

Before reproducing this experiment, make sure that:

- You have installed the ``brainlit`` package [`Documentation <https://brainlit.netlify.app/readme#installation>`_]
- You have installed PyTorch [`Documentation <https://pytorch.org/get-started/locally/>`_].

This experiment does *not* require a GPU (CUDA is not needed for PyTorch), but it would help it run faster.

If you are using `conda <https://docs.conda.io/en/latest/>`_, make sure that both ``brainlit`` and PyTorch are installed within the same environment.
