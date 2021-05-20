********
Tutorial
********

.. _utils_tutorials:

Utils
=====
Tutorials showcasing how to use the utils folder.

.. toctree::
   :maxdepth: 1

   notebooks/utils/uploading_brains
   notebooks/utils/downloading_brains
   notebooks/utils/downloading_benchmarking
   notebooks/utils/uploading_benchmarking

.. _pipeline_tutorials:

Pipelines
=========

Semi-automatic Annotation Pipeline
----------------------------------
Demonstrate pulling data and pushing traced annotations.

.. toctree::
   :maxdepth: 1

   notebooks/pipelines/seg_pipeline_demo

Segmentation
------------
Notebooks showing how to manually and automatically segment data.

.. toctree::
   :maxdepth: 1

   notebooks/pipelines/manual_segmentation
   notebooks/pipelines/tubes_feature_extraction_demo

.. _algorithm_tutorials:

Algorithms
==========

Adaptive Thresholding
---------------------
Demonstrate region growing methods using GMM and simple ITK.

.. toctree::
   :maxdepth: 1

   notebooks/algorithms/adaptive_thresh_tutorial.ipynb
   
Connecting Fragments
---------------------
Demonstrate fragment path connections using Viterbi algorithm on a simple grid example.

.. toctree::
   :maxdepth: 1

   notebooks/algorithms/viterbi_tutorial.ipynb

Soma Detection
--------------
Demonstrate simple soma detection algorithm on known somas in Janelia dataset, brain1.

.. toctree::
   :maxdepth: 1

   notebooks/algorithms/detect_somas.ipynb


.. _preprocessing_tutorials:

Preprocessing
=============

.. toctree::
   :maxdepth: 1

   notebooks/preprocessing/connectedcomponents
   notebooks/preprocessing/gaborfilter

.. _viz_tutorials:

Vizualization
=============
These tutorials demonstrate tools to load and visualize data from s3 buckets or .swc files.

.. toctree::
   :maxdepth: 1

   notebooks/visualization/loading
   notebooks/visualization/neighborhood_visualization_demo