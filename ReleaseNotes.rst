Pyshepseg Release Notes
=======================

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
