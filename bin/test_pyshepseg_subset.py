#!/usr/bin/env python

"""
Test harness for subsetting a segmented image
"""

#Copyright 2021 Neil Flood and Sam Gillingham. All rights reserved.
#
#Permission is hereby granted, free of charge, to any person 
#obtaining a copy of this software and associated documentation 
#files (the "Software"), to deal in the Software without restriction, 
#including without limitation the rights to use, copy, modify, 
#merge, publish, distribute, sublicense, and/or sell copies of the 
#Software, and to permit persons to whom the Software is furnished 
#to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be 
#included in all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
#EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
#OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
#IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR 
#ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF 
#CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION 
#WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from __future__ import print_function, division

import math
import argparse
from osgeo import gdal
gdal.UseExceptions()
from pyshepseg import tiling

DFLT_OUTPUT_DRIVER = 'KEA'
GDAL_DRIVER_CREATION_OPTIONS = {'KEA' : [], 'HFA' : ['COMPRESS=YES']}

def getCmdargs():
    """     
    Get the command line arguments.
    """
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--infile", required=True,
        help="Input file")
    p.add_argument("-o", "--outfile", required=True,
        help="Output file")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--srcwin", type=int, nargs=4,
        metavar=('xoff', 'yoff', 'xsize', 'ysize'),
        help="Top left pixel coordinates and subset size (in pixels) to extract")
    group.add_argument("--projwin", type=float, nargs=4,
        metavar=('ulx', 'uly', 'lrx', 'lry'),
        help="Projected coordinates to extract subset from the input file")
    p.add_argument("--origsegidcol", help="Name of column to write the original" +
                " segment ids")
    p.add_argument("-f", "--format", default=DFLT_OUTPUT_DRIVER, 
        choices=[DFLT_OUTPUT_DRIVER, "HFA"],
        help="Name of output GDAL format that supports RATs (default=%(default)s)")
    cmdargs = p.parse_args()
    return cmdargs
    
def getPixelCoords(fname, coords):
    """
    Open the supplied file and work out what coords (ulx, uly, lrx, lry)
    are in pixel coordinates and return (tlx, tly, xsize, ysize)
    """
    ulx, uly, lrx, lry = coords
    ds = gdal.Open(fname)
    transform = ds.GetGeoTransform()
    invTransform = gdal.InvGeoTransform(transform)
    
    pix_tlx, pix_tly = gdal.ApplyGeoTransform(invTransform, ulx, uly)
    pix_brx, pix_bry = gdal.ApplyGeoTransform(invTransform, lrx, lry)
    pix_tlx = int(pix_tlx)
    pix_tly = int(pix_tly)
    pix_brx = int(math.ceil(pix_brx))
    pix_bry = int(math.ceil(pix_bry))
    
    if (pix_tlx < 0 or pix_tly < 0 or pix_brx >= ds.RasterXSize or 
            pix_bry >= ds.RasterYSize):
        msg = 'Specified coordinates not completely within image'
        raise ValueError(msg)
        
    xsize = pix_brx - pix_tlx
    ysize = pix_bry - pix_tly
    return pix_tlx, pix_tly, xsize, ysize

def main():
    cmdargs = getCmdargs()
    
    tlx = None
    tly = None
    xsize = None
    ysize = None
    if cmdargs.srcwin is not None:
        tlx, tly, xsize, ysize = cmdargs.srcwin
    elif cmdargs.projwin is not None:
        tlx, tly, xsize, ysize = getPixelCoords(cmdargs.infile, cmdargs.projwin)
    
    creationOptions = []
    if cmdargs.format in GDAL_DRIVER_CREATION_OPTIONS:
        creationOptions = GDAL_DRIVER_CREATION_OPTIONS[cmdargs.format]
    
    tiling.subsetImage(cmdargs.infile, cmdargs.outfile, tlx, tly, 
        xsize, ysize, cmdargs.format, creationOptions, 
        cmdargs.origsegidcol)
    
if __name__ == "__main__":
    main()
