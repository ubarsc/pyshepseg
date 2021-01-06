"""

Utility functions for working with segmented data.

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

# Just in case anyone is trying to use this with Python-2
from __future__ import print_function, division

import numpy
from . import shepseg

from osgeo import gdal

DEFAULT_MINOVERVIEWDIM = 33
DEFAULT_OVERVIEWLEVELS = [ 4, 8, 16, 32, 64, 128, 256, 512 ]

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
        minVal = numpy.percentile(meanVals[1:], 5)
        maxVal = numpy.percentile(meanVals[1:], 95)
        colour = (255 * ((meanVals - minVal) / (maxVal - minVal))).clip(0, 255)
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
    
def addOverviews(ds):
    """
    Mimic rios.calcstats behaviour
    """
    # first we work out how many overviews to build based on the size
    if ds.RasterXSize < ds.RasterYSize:
        mindim = ds.RasterXSize
    else:
        mindim = ds.RasterYSize
    
    nOverviews = 0
    for i in DEFAULT_OVERVIEWLEVELS:
        if (mindim // i ) > DEFAULT_MINOVERVIEWDIM:
            nOverviews = nOverviews + 1
            
    ds.BuildOverviews("NEAREST", DEFAULT_OVERVIEWLEVELS[:nOverviews])
