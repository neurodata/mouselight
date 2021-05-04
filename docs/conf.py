# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "brainlit"
copyright = "2020, brainlit"
authors = "bvarjavand"

# The full version, including alpha/beta/rc tags
release = "0.0.0"

# -- Extension configuration -------------------------------------------------
extensions = [
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "numpydoc",
    "sphinx.ext.ifconfig",
    "sphinx.ext.githubpages",
    "sphinxcontrib.rawfiles",
    "nbsphinx",
    "sphinx.ext.intersphinx",
]

nbsphinx_kernel_name = "python3"
nbsphinx_allow_errors = True
autoapi_dirs = ["../brainlit"]
autoapi_add_toctree_entry = False
autoapi_generate_api_docs = False
nbsphinx_execute = "always"

# -- sphinxcontrib.rawfiles
# rawfiles = ["CNAME"]

# -- numpydoc
# Below is needed to prevent errors
numpydoc_show_class_members = False

# -- sphinx.ext.autosummary
autosummary_generate = True

# -- sphinx.ext.autodoc
autoclass_content = "both"
autodoc_default_flags = ["members", "inherited-members"]
autodoc_member_order = "bysource"  # default is alphabetical

# -- sphinx.ext.intersphinx
intersphinx_mapping = {
    "numpy": ("https://docs.scipy.org/doc/numpy", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "sklearn": ("http://scikit-learn.org/dev", None),
}

# -- sphinx options ----------------------------------------------------------
source_suffix = ".rst"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
master_doc = "index"
source_encoding = "utf-8"

# -- Options for HTML output -------------------------------------------------
# Add any paths that contain custom static files here, relative to this directory.
templates_path = ["_templates"]
html_static_path = ["_static"]
modindex_common_prefix = ["brainlit."]

pygments_style = "sphinx"
smartquotes = False

# Use RTD Theme
import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {
    #'includehidden': False,
    "navigation_depth": 2,
    "collapse_navigation": False,
    "navigation_depth": 3,
}
