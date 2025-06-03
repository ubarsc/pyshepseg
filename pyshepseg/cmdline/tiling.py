#!/usr/bin/env python

"""
Testing harness for running pyshepseg in the tiling mode. 
Handy for running a basic segmentation
but it is suggested that users call the module directly from a Python
script and handle things like scaling the data in an appripriate
manner for their application.

"""
# Copyright 2021 Neil Flood and Sam Gillingham. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person 
# obtaining a copy of this software and associated documentation 
# files (the "Software"), to deal in the Software without restriction, 
# including without limitation the rights to use, copy, modify, 
# merge, publish, distribute, sublicense, and/or sell copies of the 
# Software, and to permit persons to whom the Software is furnished 
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be 
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR 
# ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF 
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION 
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from __future__ import print_function, division

import sys
import time
import argparse
import json

from osgeo import gdal

from pyshepseg import tiling
from pyshepseg import tilingstats
from pyshepseg import utils

DFLT_OUTPUT_DRIVER = 'KEA'
GDAL_DRIVER_CREATION_OPTIONS = {'KEA': [], 'HFA': ['COMPRESS=YES']}

DFLT_MAX_SPECTRAL_DIFF = 'auto'

CLUSTER_CNTRS_METADATA_NAME = 'pyshepseg_cluster_cntrs'


def getCmdargs():
    """     
    Get the command line arguments.
    """
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--infile", help="Input Raster file")
    p.add_argument("-o", "--outfile")
    p.add_argument("--verbose", default=False, action="store_true",
        help="Turn on verbose output.")
    p.add_argument("--nullvalue", default=None, type=int,
        help="Null value for input image. If not given, the value set in the "+
            "image is used.")
    p.add_argument("-f", "--format", default=DFLT_OUTPUT_DRIVER, 
        choices=[DFLT_OUTPUT_DRIVER, "HFA"],
        help="Name of output GDAL format that supports RATs (default=%(default)s)")

    segGroup = p.add_argument_group("Segmentation Parameters")
    tileGroup = p.add_argument_group("Tiling Parameters")
    statsGroup = p.add_argument_group("Per-segment Statistics")
    concGroup = p.add_argument_group("Concurrency")

    segGroup.add_argument("-n", "--nclusters", default=60, type=int,
        help="Number of clusters (default=%(default)s)")
    segGroup.add_argument("--eightway", default=False, action="store_true",
        help="Use 8-way instead of 4-way")
    segGroup.add_argument("-m", "--maxspectraldiff", default=DFLT_MAX_SPECTRAL_DIFF,
        help=("Maximum Spectral Difference to use when merging " +
             "segments Either 'auto', 'none' or a value to use " +
             "(default=%(default)s)"))
    segGroup.add_argument("-s", "--minsegmentsize", default=100, type=int,
        help="Minimum segment size in pixels (default=%(default)s)")
    segGroup.add_argument("-b", "--bands", default="3,4,5", 
        help="Comma-separated list of bands to use. 1-based. (default=%(default)s)")
    segGroup.add_argument("--fixedkmeansinit", default=False, action="store_true",
        help=("Used a fixed algorithm to select guesses at cluster centres. "+
            "Default will allow the KMeans routine to select these with a "+
            "random element, which can make the final results slightly "+
            "non-determinstic. Use this if you want a completely "+
            "deterministic, reproducable result"))

    tileGroup.add_argument("-t", "--tilesize", default=tiling.DFLT_TILESIZE,
        help="Size (in pixels) of tiles to chop input image into for processing."+
                " (default=%(default)s)", type=int)
    tileGroup.add_argument("-l", "--overlapsize", default=tiling.DFLT_OVERLAPSIZE,
        help="Size (in pixels) of the overlap between tiles."+
                " (default=%(default)s)", type=int)
    tileGroup.add_argument("-c", "--clustersubsamplepercent", default=None, type=float,
        help=("Percent of data to subsample for clustering (i.e. across all "+
            "tiles). If not given, 1 million pixels are used."))
    tileGroup.add_argument("--simplerecode", default=False, action="store_true",
        help=("Use a simple recode method when merging tiles, rather "+
            "than merge segments across the tile boundary. This is mainly "+
            "for use when testing the default merge/recode. "))

    statsGroup.add_argument("--statsbands", help=("Comma-separated list of "+
        "bands in the input raster file for which to calculate per-segment "+
        "statistics, as columns in a raster attribute table (RAT). "+
        "Default will not calculate any per-segment statistics. "))
    statsGroup.add_argument("--statspec", default=[], action="append",
        help=("Specification of a statistic to be included in the "+
        "per-segment statistics in the raster attribute table (RAT). "+
        "This can be given more than once, and the nominated statistic "+
        "will be calculated for all bands given in --statsbands. "+
        "Options are 'mean', 'stddev', 'min', 'max', 'median', 'mode' or "+
        "'percentile,p' (where 'p' is a percentile (0-100) to calculate). "))
    statsGroup.add_argument("--colortablebands", help=("Comma-separated list "+
        "of 3 band numbers to use for coloring of segments. Assumes that "+
        "the per-segment mean has been calculated for each band, and this "+
        "is used to derive a colour. Band numbers are used in the order "+
        "red,green,blue"))

    concGroup.add_argument("--concurrencytype", default=tiling.CONC_NONE,
        choices=[tiling.CONC_NONE, tiling.CONC_THREADS, tiling.CONC_FARGATE,
            tiling.CONC_SUBPROC],
        help="Type of concurrency to use in tiled segmentation (default=%(default)s)")
    concGroup.add_argument("--numworkers", default=0, type=int,
        help="Number of workers for concurrent segmentation (default=%(default)s)")
    concGroup.add_argument("--fargatecfg", help=("JSON file of keyword " +
        "arguments dictionary for FargateConfig constructor " +
        "(for use with CONC_FARGATE)"))
    concGroup.add_argument("--tilecompletiontimeout", type=int, default=60,
        help=("Timeout (seconds) to wait for completion of each tile " +
              "(default=%(default)s)"))

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
    if cmdargs.statsbands is not None:
        cmdargs.statsbands = [int(x) for x in cmdargs.statsbands.split(',')]
    else:
        cmdargs.statsbands = []
    # Check that requested color table bands can be used for this. 
    if cmdargs.colortablebands is not None:
        cmdargs.colortablebands = [int(x) for x in 
            cmdargs.colortablebands.split(',')]
        if cmdargs.statspec is None or 'mean' not in cmdargs.statspec:
            print('Using --colortablebands requires "--statspec mean"')
            sys.exit()
        for i in cmdargs.colortablebands:
            if i not in cmdargs.statsbands:
                print("Bands given in --colortablebands must also be in --statsbands")
                sys.exit()

    return cmdargs


def main():
    cmdargs = getCmdargs()
    
    creationOptions = []
    if cmdargs.format in GDAL_DRIVER_CREATION_OPTIONS:
        creationOptions = GDAL_DRIVER_CREATION_OPTIONS[cmdargs.format]

    fargateCfg = None
    if cmdargs.fargatecfg is not None:
        fargateCfg_kwArgs = json.load(open(cmdargs.fargatecfg))
        fargateCfg = tiling.FargateConfig(**fargateCfg_kwArgs)
    concurrencyCfg = tiling.SegmentationConcurrencyConfig(
        concurrencyType=cmdargs.concurrencytype,
        numWorkers=cmdargs.numworkers,
        fargateCfg=fargateCfg,
        tileCompletionTimeout=cmdargs.tilecompletiontimeout)
    
    tiledSegResult = tiling.doTiledShepherdSegmentation(cmdargs.infile, cmdargs.outfile, 
            tileSize=cmdargs.tilesize, overlapSize=cmdargs.overlapsize, 
            minSegmentSize=cmdargs.minsegmentsize, numClusters=cmdargs.nclusters,
            bandNumbers=cmdargs.bands, subsamplePcnt=cmdargs.clustersubsamplepercent,
            maxSpectralDiff=cmdargs.maxspectraldiff, imgNullVal=cmdargs.nullvalue,
            fixedKMeansInit=cmdargs.fixedkmeansinit, 
            fourConnected=not cmdargs.eightway, verbose=cmdargs.verbose,
            simpleTileRecode=cmdargs.simplerecode, outputDriver=cmdargs.format,
            creationOptions=creationOptions, concurrencyCfg=concurrencyCfg)
    # Print timings
    if cmdargs.verbose and tiledSegResult.timings is not None:
        summaryDict = tiledSegResult.timings.makeSummaryDict()
        print('\n' + utils.formatTimingRpt(summaryDict) + '\n')

    # Do a colour table on final output file. 
    outDs = gdal.Open(cmdargs.outfile, gdal.GA_Update)
    band = outDs.GetRasterBand(1)

    if cmdargs.colortablebands is None:
        utils.writeRandomColourTable(band, tiledSegResult.maxSegId + 1)
    
    del outDs

    t0 = time.time()
    doPerSegmentStats(cmdargs)
    if cmdargs.verbose:
        print('Done per-segment statistics: {:.2f} seconds'.format(time.time() - t0))

    if cmdargs.colortablebands is not None:
        colorTableNames = ['Band_{}_mean'.format(i) for i in cmdargs.colortablebands]
        utils.writeColorTableFromRatColumns(cmdargs.outfile, 
            colorTableNames[0], colorTableNames[1], colorTableNames[2])


def doPerSegmentStats(cmdargs):
    """
    If requested, calculate RAT columns of per-segment statistics
    """
    for statsBand in cmdargs.statsbands:
        statsSelection = []
        for statsSpec in cmdargs.statspec:
            if statsSpec.startswith('percentile,'):
                param = int(statsSpec.split(',')[1])
                name = "Band_{}_pcnt{}".format(statsBand, param)
                selection = (name, 'percentile', param)
            else:
                name = "Band_{}_{}".format(statsBand, statsSpec)
                selection = (name, statsSpec)
            statsSelection.append(selection)

        rtn = tilingstats.calcPerSegmentStatsTiled(cmdargs.infile, statsBand,
            cmdargs.outfile, statsSelection)

        if cmdargs.verbose:
            timingsSummary = rtn.timings.makeSummaryDict()
            print(utils.formatTimingRpt(timingsSummary) + '\n')


if __name__ == "__main__":
    main()
