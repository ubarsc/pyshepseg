#!/usr/bin/env python

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
import argparse
import time

import numpy
from osgeo import gdal

from pyshepseg import shepseg

DFLT_OUTPUT_DRIVER = 'KEA'
GDAL_DRIVER_CREATION_OPTIONS = {'KEA' : [], 'HFA' : ['COMPRESS=YES']}

DFLT_MAX_SPECTRAL_DIFF = 100000

def getCmdargs():
    """     
    Get the command line arguments.
    """
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--infile", 
        help="Input Raster file")
    p.add_argument("-o", "--outfile")
    p.add_argument("-n", "--nclusters", default=30, type=int,
        help="Number of clusters (default=%(default)s)")
    p.add_argument("--subsamplepcnt", type=int, default=1,
        help="Percentage to subsample for fitting (default=%(default)s)")
    p.add_argument("--fourway", default=False, action="store_true",
        help="Use 4-way instead of 8-way")
    p.add_argument("-f", "--format", default=DFLT_OUTPUT_DRIVER, 
        choices=[DFLT_OUTPUT_DRIVER, "HFA"],
        help="Name of output GDAL format that supports RATs (default=%(default)s)")
    p.add_argument("-m", "--maxspectraldiff", default=DFLT_MAX_SPECTRAL_DIFF,
        type=int, help=("Maximum Spectral Difference to use when merging " +
                "segments (default=%(default)s)"))
        
    cmdargs = p.parse_args()
    
    if cmdargs.infile is None:
        print('Must supply input file name')
        p.print_help()
        sys.exit()

    if cmdargs.outfile is None:
        print('Must supply output file name')
        p.print_help()
        sys.exit()
        
    return cmdargs


def main():
    cmdargs = getCmdargs()
    
    t0 = time.time()
    print("Reading ... ", end='')
    ds = gdal.Open(cmdargs.infile)
    bandList = []
    imgNullVal_normed = 100
    for bn in [3, 4, 5]:
        b = ds.GetRasterBand(bn)
        refNull = b.GetNoDataValue()
        a = b.ReadAsArray()
        bandList.append(a)
    img = numpy.array(bandList)
    
    del bandList
    print(round(time.time()-t0, 1), "seconds")
    
    seg = shepseg.doShepherdSegmentation(img, 
        numClusters=60, clusterSubsamplePcnt=0.5,
        minSegmentSize=100, maxSpectralDiff=cmdargs.maxspectraldiff, 
        imgNullVal=refNull,
        fourConnected=cmdargs.fourway, verbose=True)
        
    segSize = shepseg.makeSegSize(seg)
    maxSegId = seg.max()
    spectSum = shepseg.buildSegmentSpectra(seg, img, maxSegId)

    # Write output    
    outType = gdal.GDT_UInt32
    
    (nRows, nCols) = seg.shape
    outDrvr = ds.GetDriver()
    outDrvr = gdal.GetDriverByName(cmdargs.format)
    if outDrvr is None:
        msg = 'This GDAL does not support driver {}'.format(cmdargs.format)
        raise SystemExit(msg)
    
    if os.path.exists(cmdargs.outfile):
        outDrvr.Delete(cmdargs.outfile)
    
    creationOptions = GDAL_DRIVER_CREATION_OPTIONS[cmdargs.format]
        
    outDs = outDrvr.Create(cmdargs.outfile, nCols, nRows, 1, outType,
        options=creationOptions)
    outDs.SetProjection(ds.GetProjection())
    outDs.SetGeoTransform(ds.GetGeoTransform())
    b = outDs.GetRasterBand(1)
    b.WriteArray(seg)
    b.SetMetadataItem('LAYER_TYPE', 'thematic')
    b.SetNoDataValue(shepseg.SEGNULLVAL)
    
    setColourTable(b, segSize, spectSum)
    estimateStatsFromHisto(b, segSize)
    
    del outDs


def setColourTable(bandObj, segSize, spectSum):
    """
    Set a colour table based on the segment mean spectral values. 
    It assumes we only used three bands, and assumes they will be
    mapped to (blue, green, red) in that order. 
    """
    nRows, nBands = spectSum.shape

    attrTbl = bandObj.GetDefaultRAT()
    attrTbl.SetRowCount(nRows)
    
    colNames = ["Blue", "Green", "Red"]
    colUsages = [gdal.GFU_Blue, gdal.GFU_Green, gdal.GFU_Red]
    
    for band in range(nBands):
        meanVals = spectSum[..., band] / segSize
        minVal = meanVals[1:].min()
        maxVal = meanVals[1:].max()
        colour = 255 * ((meanVals - minVal) / (maxVal - minVal))
        # reset this as it is the ignore
        colour[shepseg.SEGNULLVAL] = 0
        
        attrTbl.CreateColumn(colNames[band], gdal.GFT_Integer, colUsages[band])
        colNum = attrTbl.GetColumnCount() - 1
        attrTbl.WriteArray(colour.astype(numpy.uint), colNum)
        
    # alpha
    alpha = numpy.full((nRows,), 255, dtype=numpy.uint8)
    alpha[shepseg.SEGNULLVAL] = 0
    attrTbl.CreateColumn('Alpha', gdal.GFT_Integer, gdal.GFU_Alpha)
    colNum = attrTbl.GetColumnCount() - 1
    attrTbl.WriteArray(alpha, colNum)
    
    # histo
    # since the ignore value is shepseg.SEGNULLVAL
    # we should reset the histogram for this bin
    # so the stats are correctly calculated
    segSize[shepseg.SEGNULLVAL] = 0
    attrTbl.CreateColumn('Histogram', gdal.GFT_Integer, gdal.GFU_PixelCount)
    colNum = attrTbl.GetColumnCount() - 1
    attrTbl.WriteArray(segSize, colNum)
    
def estimateStatsFromHisto(bandObj, hist):
    """
    As a shortcut to calculating stats with GDAL, use the histogram 
    that we already have from calculating the RAT and calc the stats
    from that. 
    """
    # https://stackoverflow.com/questions/47269390/numpy-how-to-find-first-non-zero-value-in-every-column-of-a-numpy-array
    mask = hist > 0
    nVals = hist.sum()
    minVal = mask.argmax()
    maxVal = hist.shape[0] - numpy.flip(mask).argmax() - 1
    
    values = numpy.arange(hist.shape[0])
    
    meanVal = (values * hist).sum() / nVals
    
    stdDevVal = (hist * numpy.power(values - meanVal, 2)).sum() / nVals
    stdDevVal = numpy.sqrt(stdDevVal)
    
    modeVal = numpy.argmax(hist)
    # estimate the median - bin with the middle number
    middlenum = hist.sum() / 2
    gtmiddle = hist.cumsum() >= middlenum
    medianVal = gtmiddle.nonzero()[0][0]
    
    bandObj.SetMetadataItem("STATISTICS_MINIMUM", repr(minVal))
    bandObj.SetMetadataItem("STATISTICS_MAXIMUM", repr(maxVal))
    bandObj.SetMetadataItem("STATISTICS_MEAN", repr(meanVal))
    bandObj.SetMetadataItem("STATISTICS_STDDEV", repr(stdDevVal))
    bandObj.SetMetadataItem("STATISTICS_MODE", repr(modeVal))
    bandObj.SetMetadataItem("STATISTICS_MEDIAN", repr(medianVal))
    bandObj.SetMetadataItem("STATISTICS_SKIPFACTORX", "1")
    bandObj.SetMetadataItem("STATISTICS_SKIPFACTORY", "1")
    bandObj.SetMetadataItem("STATISTICS_HISTOBINFUNCTION", "direct")
    
if __name__ == "__main__":
    main()
