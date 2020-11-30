.. role:: raw-html-m2r(raw)
   :format: html


Brainlit
========


.. image:: https://travis-ci.com/neurodata/brainlit.svg?branch=master
   :target: https://travis-ci.com/neurodata/brainlit
   :alt: Build Status


.. image:: https://badge.fury.io/py/brainlit.svg
   :target: https://badge.fury.io/py/brainlit
   :alt: PyPI version


.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: black


.. image:: https://codecov.io/gh/neurodata/brainlit/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/neurodata/brainlit
   :alt: codecov


.. image:: https://img.shields.io/docker/cloud/build/bvarjavand/brainlit
   :target: https://img.shields.io/docker/cloud/build/bvarjavand/brainlit
   :alt: Docker Cloud Build Status


.. image:: https://img.shields.io/docker/image-size/bvarjavand/brainlit
   :target: https://img.shields.io/docker/image-size/bvarjavand/brainlit
   :alt: Docker Image Size (latest by date)


.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
   :alt: License
   

This repository is a container of methods that Neurodata usees to expose their open-source code while it is in the process of being merged with larger scientific libraries such as scipy, scikit-image, or scikit-learn. Additionally, methods for computational neuroscience on brains too specific for a general scientific library can be found here, such as image registration software tuned specifically for large brain volumes.


.. image:: https://i.postimg.cc/QtG9Xs68/Brainlit.png
   :target: https://i.postimg.cc/QtG9Xs68/Brainlit.png
   :alt: Brainlight Features


.. toctree::
   :numbered:


Motivation
----------

The repository originated as the project of a team in Joshua Vogelstein's class **Neurodata** at Johns Hopkins University. This project was focused on data science towards the `mouselight data <https://www.hhmi.org/news/mouselight-project-maps-1000-neurons-and-counting-in-the-mouse-brain>`_. It becme apparent that the tools developed for the class would be useful for other groups doing data science on large data volumes.
The repository can now be considered a "holding bay" for code developed by Neurodata for collaborators and researchers to use.

Installation
------------

Environment
^^^^^^^^^^^

(optional, any python >= 3.8 environment will suffice)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* `get conda <https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html>`_
* create a virtual environment: ``conda create --name brainlit python=3.8``
* activate the environment: ``conda activate brainlit``

Install from pypi
^^^^^^^^^^^^^^^^^


* install brainlit: ``pip install brainlit``

Install from source
^^^^^^^^^^^^^^^^^^^


* clone the repo: ``git clone https://github.com/neurodata/brainlit.git``
* cd into the repo: ``cd brainlit``
* install brainlit: ``pip install -e .``

How to use Brainlit
-------------------

Data setup
^^^^^^^^^^

The ``source`` data directory should have an octree data structure

.. code-block::

    data/
   ├── default.0.tif
   ├── transform.txt
   ├── 1/
   │   ├── 1/, ..., 8/
   │   └── default.0.tif
   ├── 2/ ... 8/
   └── consensus-swcs (optional)
       ├── G-001.swc
       ├── G-002.swc
       └── default.0.tif

If your team wants to interact with cloud data, each member will need account credentials specified in ``~/.cloudvolume/secrets/x-secret.json``\ , where ``x`` is one of ``[aws, gc, azure]`` which contains your id and secret key for your cloud platform.
We provide a template for ``aws`` in the repo for convenience.

Create a session
^^^^^^^^^^^^^^^^

Each user will start their scripts with approximately the same lines:

.. code-block::

   from brainlit.utils.ngl import NeuroglancerSession

   session = NeuroglancerSession(url='file:///abc123xyz')

From here, any number of tools can be run such as the visualization or annotation tools. `Viz demo <https://github.com/neurodata/brainlit/blob/master/docs/notebooks/visualization/visualization.ipynb>`_.

Features
--------

Registration
^^^^^^^^^^^^

The registration subpackage is a facsimile of ARDENT, a pip-installable (pip install ardent) package for nonlinear image registration wrapped in an object-oriented framework for ease of use. This is an implementation of the LDDMM algorithm with modifications, written by Devin Crowley and based on "Diffeomorphic registration with intensity transformation and missing data: Application to 3D digital pathology of Alzheimer's disease." This paper extends on an older LDDMM paper, "Computing large deformation metric mappings via geodesic flows of diffeomorphisms."

This is the more recent paper:

Tward, Daniel, et al. "Diffeomorphic registration with intensity transformation and missing data: Application to 3D digital pathology of Alzheimer's disease." Frontiers in neuroscience 14 (2020).

https://doi.org/10.3389/fnins.2020.00052

This is the original LDDMM paper:

Beg, M. Faisal, et al. "Computing large deformation metric mappings via geodesic flows of diffeomorphisms." International journal of computer vision 61.2 (2005): 139-157.

https://doi.org/10.1023/B:VISI.0000043755.93987.aa

A tutorial is available in docs/notebooks/registration_demo.ipynb.

Core
----

The core brain-lit package can be described by the diagram at the top of the readme:

(Push and Pull Data)
^^^^^^^^^^^^^^^^^^^^

Brainlit uses the Seung Lab's `Cloudvolume <https://github.com/seung-lab/cloud-volume>`_ package to push and pull data through the cloud or a local machine in an efficient and parallelized fashion. `Uploading demo <https://github.com/neurodata/brainlit/blob/master/docs/notebooks/utils/uploading_brains.ipynb>`_.\ :raw-html-m2r:`<br>`
The only requirement is to have an account on a cloud service on s3, azure, or google cloud.

Loading data via local filepath of an octree structure is also supported. `Octree demo <https://github.com/neurodata/brainlit/blob/master/docs/notebooks/utils/upload_brains.ipynb>`_.

Visualize
^^^^^^^^^

Brainlit supports many methods to visualize large data. Visualizing the entire data can be done via Google's `Neuroglancer <https://github.com/google/neuroglancer>`_\ , which provides a web link as shown below.

screenshot

Brainlit also has tools to visualize chunks of data as 2d slices or as a 3d model. `Visualization demo <https://github.com/neurodata/brainlit/blob/master/docs/notebooks/visualization/visualization.ipynb>`_.

screenshot

Manually Segment
^^^^^^^^^^^^^^^^

Brainlit includes a lightweight manual segmentation pipeline. This allows collaborators of a projec to pull data from the cloud, create annotations, and push their annotations back up as a separate channel. `Auto demo <https://github.com/neurodata/brainlit/blob/master/docs/notebooks/pipelines/manual_segementation.ipynb>`_.

Automatically and Semi-automatically Segment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Similar to the above pipeline, segmentations can be automatically or semi-automatically generated and pushed to a separate channel for viewing. `Semi-auto demo <https://github.com/neurodata/brainlit/blob/master/docs/notebooks/pipelines/seg_pipeline_demo.ipynb>`_.

API Reference
-------------


.. image:: https://readthedocs.org/projects/brainlight/badge/?version=latest
   :target: https://brainlight.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

The documentation can be found at `https://brainlight.readthedocs.io/en/latest/ <https://brainlight.readthedocs.io/en/latest/>`_.

Tests
-----

Running tests can easily be done by moving to the root directory of the brainlit package ant typing ``pytest tests`` or ``python -m pytest tests``.\ :raw-html-m2r:`<br>`
Running a specific test, such as ``test_upload.py`` can be done simply by ``ptest tests/test_upload.py``.

Contributing
------------

Contribution guidelines can be found via `CONTRIBUTING.md <https://github.com/neurodata/brainlit/blob/master/CONTRIBUTING.md>`_

Credits
-------

Thanks to the neurodata team and the group in the neurodata class which started the project.
This project is currently managed by Tommy Athey and Bijan Varjavand.
