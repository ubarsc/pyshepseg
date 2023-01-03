.. pyshepseg documentation master file, created by
   sphinx-quickstart on Mon Dec  6 11:34:41 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyshepseg
=========

Introduction
------------
Python implementation of image segmentation algorithm of 
Shepherd et al (2019). *Operational Large-Scale Segmentation of Imagery Based on Iterative 
Elimination*. `Remote Sensing 11(6) <https://www.mdpi.com/2072-4292/11/6/658>`_.

This package is a tool for Python programmers to implement the segmentation algorithm. 
It is not a stand-alone solution for people with no Python experience.

We thank the authors of the paper for their algorithm. This implementation was created 
independently of them, and they are in no way to blame for any mistakes we have made.

Downloads
---------

From `GitHub <https://github.com/ubarsc/pyshepseg/releases>`_. 
Release notes by version can be viewed in :doc:`ReleaseNotes`.

Dependencies
------------
The package requires the `scikit-learn <https://scikit-learn.org/>`_ package,
and the `numba <https://numba.pydata.org/>`_ package. These need to be installed
before this package will run. See their instructions on how to install, and
choose whichever methods best suits you. Both packages are available in 
mutually consistent builds from the conda-forge archive, but many other 
options are available. 

Also recommended is the `GDAL <https://gdal.org/>`_ package for reading and 
writing raster file formats. It is not required by the core segmentation
algorithm, but is highly recommended as a portable way to interface 
to a large range of raster formats. It is required by the ``tiling`` module
to support segmentation of large rasters. The GDAL package is also available 
from conda-forge, but again, other installation options are available. 

Installation
------------
The package can be installed directly from the source, using the 
setup.py script. 

1. The source code is available from `<https://github.com/ubarsc/pyshepseg>`_.
   Either unpack the latest release bundle from 
   `<https://github.com/ubarsc/pyshepseg/releases>`_, or clone the 
   repository. 
2. Run the setup.py script. This is best done by using pip as a wrapper
   around it, as follows. Note that pip has a ``--prefix`` option to allow
   installation in non-standard locations.

::

   pip install .


Usage
-----

::

  from pyshepseg import shepseg

  # Read in a multi-band image as a single array, img,
  # of shape (nBands, nRows, nCols). 
  # Ensure that any null pixels are all set to a known 
  # null value in all bands. Failure to correctly identify 
  # null pixels can result in a poorer quality segmentation. 

  segRes = shepseg.doShepherdSegmentation(img, imgNullVal=nullVal)


The segimg attribute of the segRes object is an array
of segment ID numbers, of shape (nRows, nCols). 

See the help in the shepseg module for further details and tips. 

Large Rasters
-------------
The basic usage outlined above operates entirely in-memory. For
very large rasters, this can be infeasible. A tiled implementation
is provided in the ``pyshepseg.tiling`` module, which divides a large 
raster into fixed-size tiles, segments each tile in-memory, and 
stitched the results together to create a single segment image. The 
tiles are stitched such that segments are matched and merged across 
tile boundaries, so the result is seamless. 

This technique should be used with caution. See the docstring for
the ``pyshepseg.tiling`` module for further discussion of usage and
caveats. 

Once a segmentation has been completed, statistics can be gathered per segment on
large rasters using the functions defined in the ``pyshepseg.tilingstats``
module.

Command Line Scripts
--------------------
A few basic command line scripts are also provided as entry points.
Their main purpose is as test scripts during development, but they also serve 
as examples of how to write scripts which use the package. In addition, 
they can also be used directly for simple segmentation tasks. 

The ``pyshepseg_run_seg`` entry point is a wrapper around the basic in-memory usage.

The ``pyshepseg_tiling`` entry point is a wrapper around the tiled
segmentation for large rasters. 

The ``pyshepseg_subset`` entry point uses the ``subset.subsetImage``
function to subset a segmentation image, re-labelling the segments
to contiguous segment ID numbers. 

The ``pyshepseg_variograms`` entry point uses the
``tilingstats.calcPerSegmentSpatialStatsTiled`` function to calculate the
given number of variograms.

The ``pyshepseg_runtests`` entry point runs some tests on packages data and
can be used to confirm that the behaviour of this package is as expected.

Use the ``--help`` option on each script for usage details. 

Per-segment Statistics
----------------------
It can be useful to calculate statistics of the pixels from 
the original input imagery on a per-segment basis. For example, for
all the pixels in a single segment, one might calculate the mean value 
of one or more of the bands from the original imagery. 

A routine is provided to do this in a memory-efficient way, given the
original image and the completed segmentation image. A standard set of
statistics are available, including mean, standard deviation, and 
arbitrary percentile values, amongst others. The selected per-segment 
statistics are written to the segment image file as columns of a raster
attribute table (RAT). 

For details, see the help on the ``tilingstats.calcPerSegmentStatsTiled()``
function. 

Segment Colour Tables
---------------------
The segment image contains a large number of individual segment values, and 
can be difficult to view in simple greyscale colouring. To improve this, two 
routines are provided in the ``pyshepseg.utils`` module which will attach a colour table. 

The simplest routine is ``utils.writeRandomColourTable``, which attaches a 
randomly-generated colour table, so that each segment is assigned a randomly 
chosen colour, which merely serves to distinguish it from the surrounding segments. 
See its help for details. 

More sophisticated and more useful is the function ``utils.writeColorTableFromRatColumns``,
which uses previously calculated columns in the raster attribute table (RAT) to 
create a colour table which approximates the original imagery. See its help for 
details, and the preceding section on how to create suitable RAT columns. 

Modules in this Package
=======================

.. toctree::
   :maxdepth: 1
   
   pyshepseg_shepseg
   pyshepseg_tiling
   pyshepseg_tilingstats
   pyshepseg_utils
   pyshepseg_subset


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
