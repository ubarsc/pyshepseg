#!/usr/bin/env python

"""
Testing harness for pyshepseg. Handy for running a basic segmentation
but it is suggested that users call the module directly from a Python
script and handle things like scaling the data in an appripriate
manner for their application.

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

import os
import sys
import json
import argparse
import time

import numpy
from osgeo import gdal

from pyshepseg import shepseg
from pyshepseg import utils

DFLT_OUTPUT_DRIVER = 'KEA'
GDAL_DRIVER_CREATION_OPTIONS = {'KEA' : [], 'HFA' : ['COMPRESS=YES']}

DFLT_MAX_SPECTRAL_DIFF = 'auto'

CLUSTER_CNTRS_METADATA_NAME = 'pyshepseg_cluster_cntrs'

def getCmdargs():
    """     
    Get the command line arguments.
    """
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--infile", help="Input Raster file")
    p.add_argument("-o", "--outfile")
    p.add_argument("-n", "--nclusters", default=60, type=int,
        help="Number of clusters (default=%(default)s)")
    p.add_argument("--subsamplepcnt", type=int, default=1,
        help="Percentage to subsample for fitting (default=%(default)s)")
    p.add_argument("--eightway", default=False, action="store_true",
        help="Use 8-way instead of 4-way")
    p.add_argument("-f", "--format", default=DFLT_OUTPUT_DRIVER, 
        choices=[DFLT_OUTPUT_DRIVER, "HFA"],
        help="Name of output GDAL format that supports RATs (default=%(default)s)")
    p.add_argument("-m", "--maxspectraldiff", default=DFLT_MAX_SPECTRAL_DIFF,
        help=("Maximum Spectral Difference to use when merging " +
                "segments Either 'auto', 'none' or a value to use " +
                "(default=%(default)s)"))
    p.add_argument("-s", "--minsegmentsize", default=100, type=int,
        help="Minimum segment size in pixels (default=%(default)s)")
    p.add_argument("-c", "--clustersubsamplepercent", default=0.5, type=float,
        help="Percent of data to subsample for clustering (default=%(default)s)")
    p.add_argument("-b", "--bands", default="3,4,5", 
        help="Comma seperated list of bands to use. 1-based. (default=%(default)s)")
    p.add_argument("--fixedkmeansinit", default=False, action="store_true",
        help=("Used a fixed algorithm to select guesses at cluster centres. "+
            "Default will allow the KMeans routine to select these with a "+
            "random element, which can make the final results slightly "+
            "non-determinstic. Use this if you want a completely "+
            "deterministic, reproducable result"))
        
    cmdargs = p.parse_args()
    
    if cmdargs.infile is None:
        print('Must supply input file name')
        p.print_help()
        sys.exit()

    if cmdargs.outfile is None:
        print('Must supply output file name')
        p.print_help()
        sys.exit()
        
    try:
        cmdargs.maxspectraldiff = float(cmdargs.maxspectraldiff)
    except ValueError:
        # check for 'auto' or 'none'
        if cmdargs.maxspectraldiff not in ('auto', 'none'):
            print("Only 'auto', 'none' or a value supported for --maxspectraldiff")
            p.print_help()
            sys.exit()
          
        # code expects 'none' -> None
        if cmdargs.maxspectraldiff == 'none':
            cmdargs.maxspectraldiff = None
            
    # turn string of bands into list of ints
    cmdargs.bands = [int(x) for x in cmdargs.bands.split(',')]
            
    return cmdargs


def main():
    cmdargs = getCmdargs()
    
    t0 = time.time()
    print("Reading ... ", end='')
    (img, refNull) = readImageBands(cmdargs)
    print(round(time.time()-t0, 1), "seconds")
    
    # Do the segmentation
    segResult = shepseg.doShepherdSegmentation(img, 
        numClusters=cmdargs.nclusters, 
        clusterSubsamplePcnt=cmdargs.clustersubsamplepercent,
        minSegmentSize=cmdargs.minsegmentsize, 
        maxSpectralDiff=cmdargs.maxspectraldiff, 
        imgNullVal=refNull, fourConnected=not cmdargs.eightway, 
        fixedKMeansInit=cmdargs.fixedkmeansinit, verbose=True)

    # The segmentation image, and a few related quantities    
    seg = segResult.segimg
    segSize = shepseg.makeSegSize(seg)
    maxSegId = seg.max()
    spectSum = shepseg.buildSegmentSpectra(seg, img, maxSegId)
    kmeansObj = segResult.kmeans

    writeOutput(cmdargs, seg, segSize, spectSum, kmeansObj)


def writeOutput(cmdargs, seg, segSize, spectSum, kmeansObj):
    """
    Write the segmentation to an output image file. Includes a 
    colour table
    """
    # Write output    
    outType = gdal.GDT_UInt32
    
    (nRows, nCols) = seg.shape
    outDrvr = gdal.GetDriverByName(cmdargs.format)
    if outDrvr is None:
        msg = 'This GDAL does not support driver {}'.format(cmdargs.format)
        raise SystemExit(msg)
    
    if os.path.exists(cmdargs.outfile):
        outDrvr.Delete(cmdargs.outfile)
    
    creationOptions = GDAL_DRIVER_CREATION_OPTIONS[cmdargs.format]
    
    inDs = gdal.Open(cmdargs.infile)
        
    outDs = outDrvr.Create(cmdargs.outfile, nCols, nRows, 1, outType,
        options=creationOptions)
    outDs.SetProjection(inDs.GetProjection())
    outDs.SetGeoTransform(inDs.GetGeoTransform())
    b = outDs.GetRasterBand(1)
    b.WriteArray(seg)
    b.SetMetadataItem('LAYER_TYPE', 'thematic')
    b.SetNoDataValue(shepseg.SEGNULLVAL)
    
    # since we have the histo we can do the stats
    utils.estimateStatsFromHisto(b, segSize)
    
    # overviews
    utils.addOverviews(outDs)
    
    # save the cluster centres
    writeClusterCentresToMetadata(b, kmeansObj)
    
    del outDs


def readImageBands(cmdargs):
    """
    Read in the requested bands of the given image. Return
    a tuple of the image array and the null value.
    """
    ds = gdal.Open(cmdargs.infile)
    bandList = []
    for bn in cmdargs.bands:
        b = ds.GetRasterBand(bn)
        refNull = b.GetNoDataValue()
        a = b.ReadAsArray()
        bandList.append(a)
    img = numpy.array(bandList)
    
    return (img, refNull)


def writeClusterCentresToMetadata(bandObj, km):
    """
    Pulls out the cluster centres from the kmeans object 
    and writes them to the metadata (under CLUSTER_CNTRS_METADATA_NAME)
    for the given band object.
    """
    # convert to list so we can json them
    ctrsList = [list(r) for r in km.cluster_centers_]
    ctrsString = json.dumps(ctrsList)
    
    bandObj.SetMetadataItem(CLUSTER_CNTRS_METADATA_NAME, ctrsString)


if __name__ == "__main__":
    main()
