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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'pyshepseg'
copyright = '2021, Neil Flood & Sam Gillingham'
author = 'Neil Flood & Sam Gillingham'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon'
]
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'classic'
html_theme_options = {
    "sidebarwidth": "20%",
    "body_min_width": "90%",
    "stickysidebar": True
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# Set up list of things to mock, if they are not actually present. In other
# words, don't mock them when they are present. This is mainly to avoid
# a list of warning messages coming from Sphinx while testing, but
# makes no real difference when running on ReadTheDocs (when everything
# will be mocked anyway).
# I am very unsure about this. I would much prefer to remove the warnings
# for real, but don't yet know how.
possibleMockList = ['numpy', 'numba', 'osgeo', 'scipy', 'sklearn']
autodoc_mock_imports = []
for module in possibleMockList:
    try:
        exec('import {}'.format(module))
    except ImportError:
        autodoc_mock_imports.append(module)
