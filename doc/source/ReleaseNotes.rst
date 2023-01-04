Pyshepseg Release Notes
=======================

Version 2.0.0 (2023-01-04)
--------------------------

New Features:
  * A test script (pyshepseg_runtests) that can be run to confirm 
    the install is working as intended.
  * Split up the parts of doTiledSegmentation() so they can be run
    in parallel.
  * Check syntax with flake8 and run test script on new PRs in github.
  * Use entry points for the command line scripts rather than creating
    our own. Should make running on Windows easier.
  * Added ability to calculate "spatial" statistics on the segments.
  * Use numpydoc for creating Sphinx documentation.
  * Subset functionality is now in a separate "subset" module.
  * Statistics functionality now in a new "tilingstats" module.

Version 1.1.0 (2021-12-24)
--------------------------

Bug Fixes:
  * Guard against Float images being used for calculating
    statistics as the results were undefined.
  * Added other checks to ensure that the image having statistics
    calculated matches spatially with the segmented image.
  * Add the ability to add GDAL driver creation options for the
    output image of a segmentation.
  * Create the histogram column as a Real to match common GDAL 
    usage.
  * Add checks to ensure histogram and colour columns aren't
    created if they already exist.
  * Ensure the first segment of each RAT Page isn't initally set
    to 'complete' before use.
  * Raise error if any incomplete RAT Pages are found during processing
    as this indicates the RAT contains more entries than unique values
    in the image.
  * When calculating statistics, open the file that the stats are
    calculated on in read only mode so /vsi filesystems can be used.
  * Increase default overlap for tiled segmentation as the old value
    could result in inconsistencies and segments that were missing from
    the image (but in the RAT).
  * Remove dependency on distutils which is now deprecated.

New Features:
  * New Sphinx documentation located at https://www.pyshepseg.org.
  * Added a new subsetImage() function to the tiling module that subsets
    an already segemented image and collapses the RAT so there are no
    redundant entries. Also added a test_pyshepseg_subset.py command line
    program to test this functionality.
  * Exclude any nodata pixels values during statistics calculation.

Version 1.0.0 (2021-04-08)
--------------------------

New Features:
  * Added pyshsep.tiling module to allow processing of large rasters
    in a memory-efficient manner. 
  * Added pyshepseg.tiling.calcPerSegmentStatsTiled() function to 
    enable calculation of per-segment statistics in a fast and 
    memory-efficient manner. 
  * Added pyshepseg.utils.writeColorTableFromRatColumns() function, to
    add colour table calculated from per-segment statistics

Version 0.1 
-----------

Initial implementation of segmentation algorithm. Other facilities
will be added as we get to them. 
