#!/usr/bin/env python

"""
Testing harness for running pyshepseg in the tiling mode. 
Handy for running a basic segmentation
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
import time
import argparse

from osgeo import gdal

from pyshepseg import tiling
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
    p.add_argument("-t", "--tilesize", default=tiling.DFLT_TILESIZE,
        help="Size (in pixels) of tiles to chop input image into for processing."+
                " (default=%(default)s)", type=int)
    p.add_argument("-l", "--overlapsize", default=tiling.DFLT_OVERLAPSIZE,
        help="Size (in pixels) of the overlap between tiles."+
                " (default=%(default)s)", type=int)
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
    p.add_argument("-c", "--clustersubsamplepercent", default=None, type=float,
        help="Percent of data to subsample for clustering. If not given, "+
            "1 million pixels are used.")
    p.add_argument("-b", "--bands", default="3,4,5", 
        help="Comma seperated list of bands to use. 1-based. (default=%(default)s)")
    p.add_argument("--nullvalue", default=None, type=int,
        help="Null value for input image. If not given, the value set in the "+
            "image is used.")
    p.add_argument("--fixedkmeansinit", default=False, action="store_true",
        help=("Used a fixed algorithm to select guesses at cluster centres. "+
            "Default will allow the KMeans routine to select these with a "+
            "random element, which can make the final results slightly "+
            "non-determinstic. Use this if you want a completely "+
            "deterministic, reproducable result"))
    p.add_argument("--verbose", default=False, action="store_true",
        help="Turn on verbose output.")
    p.add_argument("--simplerecode", default=False, action="store_true",
        help="Use a simple recode method rather than merge segments on the overlap")
        
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
    
    maxSegId = tiling.doTiledShepherdSegmentation(cmdargs.infile, cmdargs.outfile, 
            tileSize=cmdargs.tilesize, overlapSize=cmdargs.overlapsize, 
            minSegmentSize=cmdargs.minsegmentsize, numClusters=cmdargs.nclusters,
            bandNumbers=cmdargs.bands, subsamplePcnt=cmdargs.clustersubsamplepercent,
            maxSpectralDiff=cmdargs.maxspectraldiff, imgNullVal=cmdargs.nullvalue,
            fixedKMeansInit=cmdargs.fixedkmeansinit, 
            fourConnected=not cmdargs.eightway, verbose=cmdargs.verbose,
            simpleTileRecode=cmdargs.simplerecode, outputDriver=cmdargs.format)

    # Do histogram, stats and colour table on final output file. 
    outDs = gdal.Open(cmdargs.outfile, gdal.GA_Update)

    t0 = time.time()
    hist = tiling.calcHistogramTiled(outDs, maxSegId, writeToRat=True)
    if cmdargs.verbose:
        print('Done histogram: {:.2f} seconds'.format(time.time()-t0))

    band = outDs.GetRasterBand(1)

    utils.estimateStatsFromHisto(band, hist)
    utils.writeRandomColourTable(band, maxSegId+1)
    
    del outDs    

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
