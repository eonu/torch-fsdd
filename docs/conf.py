# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import sys, os

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
try:
    # TorchFSDD is installed
    import torchfsdd
except ImportError:
    # TorchFSDD is run from its source checkout
    sys.path.insert(0, os.path.abspath('../lib'))
    import torchfsdd

# -- Project information -----------------------------------------------------

project = torchfsdd.__name__
author = torchfsdd.__author__
copyright = torchfsdd.__copyright__
release = torchfsdd.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'numpydoc',
    'm2r2',
    'sphinx.ext.intersphinx',
    'versionwarning.extension'
]

intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'torchaudio': ('https://pytorch.org/audio/stable/', None),
    'torchvision': ('https://pytorch.org/vision/stable/', None),
    'python': ('http://docs.python.org/dev', None)
}

autodoc_member_order = 'bysource'
autosummary_generate = True
numpydoc_show_class_members = False

# Set master document
master_doc = 'index'

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = ['.rst', '.md']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']