# pyshepseg
Python implementation of image segmentation algorithm of 
[Shepherd et al (2019) Operational Large-Scale Segmentation of Imagery 
Based on Iterative Elimination. Remote Sensing 11(6).](https://www.mdpi.com/2072-4292/11/6/658) 

## Dependencies
The package requires the [scikit-learn](https://scikit-learn.org/) package,
and the [numba](https://numba.pydata.org/) package. These need to be installed
before this package will run. See their instructions on how to install, and
choose whichever methods best suits you. 

Also recommended is the [GDAL](https://gdal.org/) package for reading and 
writing raster file formats. It is not required by the current package, 
but is highly recommended as a portable way to interface 
to a large range of raster formats. 

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

