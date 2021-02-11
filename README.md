# pyshepseg
Python implementation of image segmentation algorithm of 
Shepherd et al (2019). Operational Large-Scale Segmentation of Imagery 
Based on Iterative Elimination. [Remote Sensing 11(6).](https://www.mdpi.com/2072-4292/11/6/658) 

This package is a tool for Python programmers to implement the segmentation 
algorithm. It is not a stand-alone solution for people with no Python
experience. 

We thank the authors of the paper for their algorithm. This implementation 
was created independently of them, and they are in no way to blame for 
any mistakes we have made. 

## Dependencies
The package requires the [scikit-learn](https://scikit-learn.org/) package,
and the [numba](https://numba.pydata.org/) package. These need to be installed
before this package will run. See their instructions on how to install, and
choose whichever methods best suits you. Both packages are available in 
mutually consistent builds from the conda-forge archive, but many other 
options are available. 

Also recommended is the [GDAL](https://gdal.org/) package for reading and 
writing raster file formats. It is not required by the core segmentation
algorithm, but is highly recommended as a portable way to interface 
to a large range of raster formats. It is required by the `tiling` module
to support segmentation of large rasters. The GDAL package is also available 
from conda-forge, but again, other installation options are available. 

## Installation
The package can be installed directly from the source, using the 
setup.py script. 

1. Clone the github pyshepseg repository
2. Run the setup.py script, e.g.
```
python setup.py install
```

## Usage

```
  from pyshepseg import shepseg

  # Read in a multi-band image as a single array, img,
  # of shape (nBands, nRows, nCols). 
  # Ensure that any null pixels are all set to a known 
  # null value in all bands. Failure to correctly identify 
  # null pixels can result in a poorer quality segmentation. 

  segRes = shepseg.doShepherdSegmentation(img, imgNullVal=nullVal)
```

The segimg attribute of the segRes object is an array
of segment ID numbers, of shape (nRows, nCols). 

See the help in the shepseg module for further details and tips. 

### Large Rasters
The basic usage outlined above operates entirely in-memory. For
very large rasters, this can be infeasible. A tiled implementation
is provided in the `pyshepseg.tiling` module, which divides a large 
raster into fixed-size tiles, segments each tile in-memory, and 
stitched the results together to create a single segment image. The 
tiles are stitched such that segments are matched and merged across 
tile boundaries, so the result is seamless. 

This technique should be used with caution. See the docstring for
the `pyshepseg.tiling` module for further discussion of usage and
caveats. 

### Command Line Scripts
Two basic command line scripts are also provided in the `bin/` subdirectory. 
Their main purpose is as test scripts during development, but they also serve 
as examples of how to write scripts which use the package. In addition, 
they can also be used directly for simple segmentation tasks. 

The `test_pyshepseg.py` script is a wrapper around the basic in-memory usage. 
The `test_pyshepseg_tiling.py` script is a wrapper around the tiled 
segmentation for large rasters. 

