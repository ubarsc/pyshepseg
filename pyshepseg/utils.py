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

DEFAULT_MINOVERVIEWDIM = 100
DEFAULT_OVERVIEWLEVELS = [ 4, 8, 16, 32, 64, 128, 256, 512 ]

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


def writeRandomColourTable(outBand, nRows):
    """
    Attach a randomly-generated colour table to the given segmentation
    image. The image is given as an open gdal.Band object. Also
    requires the number of rows in the attribute table, i.e. the
    number of segments + 1. 
    
    """
    nRows = int(nRows)
    colNames = ["Blue", "Green", "Red"]
    colUsages = [gdal.GFU_Blue, gdal.GFU_Green, gdal.GFU_Red]

    attrTbl = outBand.GetDefaultRAT()
    attrTbl.SetRowCount(nRows)
    
    for band in range(3):
        colNum = attrTbl.GetColOfUsage(colUsages[band])
        if colNum == -1:
            attrTbl.CreateColumn(colNames[band], gdal.GFT_Integer, colUsages[band])
            colNum = attrTbl.GetColumnCount() - 1
        colour = numpy.random.random_integers(0, 255, size=nRows)
        attrTbl.WriteArray(colour, colNum)
        
    alpha = numpy.full((nRows,), 255, dtype=numpy.uint8)
    alpha[shepseg.SEGNULLVAL] = 0
    attrTbl.CreateColumn('Alpha', gdal.GFT_Integer, gdal.GFU_Alpha)
    colNum = attrTbl.GetColumnCount() - 1
    attrTbl.WriteArray(alpha, colNum)


def writeColorTableFromRatColumns(segfile, redColName, greenColName, 
        blueColName):
    """
    Use the values in the given columns in the raster attribute 
    table (RAT) to create corresponding color table columns, so that 
    the segmented image will display similarly to same bands of the 
    the original image. 

    segfile is the completed segmentation image, with RAT columns 
    already written. 

    The three column names are strings, being the names of columns 
    to use to derive corresponding colors. 

    The general idea is that the given columns would be the per-segment 
    means of the desired bands (see tiling.calcPerSegmentStatsTiled()
    to create such columns). 
    """
    colList = [redColName, greenColName, blueColName]
    colorColList = ['Red', 'Green', 'Blue']
    usageList = [gdal.GFU_Red, gdal.GFU_Green, gdal.GFU_Blue]

    ds = gdal.Open(segfile, gdal.GA_Update)
    band = ds.GetRasterBand(1)
    attrTbl = band.GetDefaultRAT()
    colNameList = [attrTbl.GetNameOfCol(i) 
        for i in range(attrTbl.GetColumnCount())]

    for i in range(3):
        n = colNameList.index(colList[i])
        colVals = attrTbl.ReadAsArray(n)

        # If the corresponding color column does not yet exist, then create it
        if colorColList[i] not in colNameList:
            attrTbl.CreateColumn(colorColList[i], gdal.GFT_Integer, usageList[i])
            clrColNdx = attrTbl.GetColumnCount() - 1
        else:
            clrColNdx = colNameList.index(colorColList[i])

        # Use the column values to create a color column of values in 
        # the range 0-255. Stretch to the 5-th and 95th percentiles, to
        # avoid extreme values causing washed out colors. 
        colMin = numpy.percentile(colVals, 5)
        colMax = numpy.percentile(colVals, 95)
        clr = (255 * ((colVals - colMin) / (colMax - colMin)).clip(0, 1))

        # Write the color column
        attrTbl.WriteArray(clr.astype(numpy.uint8), clrColNdx)

    # Now write the opacity (alpha) column. Set to full opacity. 
    alpha = numpy.full(len(colVals), 255, dtype=numpy.uint8)
    if 'Alpha' not in colNameList:
        attrTbl.CreateColumn('Alpha', gdal.GFT_Integer, gdal.GFU_Alpha)
        i = attrTbl.GetColumnCount() - 1
    else:
        i = colNameList.index('Alpha')
    attrTbl.WriteArray(alpha, i)
