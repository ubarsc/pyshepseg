"""

Utility functions for working with segmented data.

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

# Just in case anyone is trying to use this with Python-2
from __future__ import print_function, division

import sys
import inspect
import traceback

import numpy
from . import shepseg

from osgeo import gdal

gdal.UseExceptions()

DEFAULT_MINOVERVIEWDIM = 100
DEFAULT_OVERVIEWLEVELS = [4, 8, 16, 32, 64, 128, 256, 512]
gdalFloatTypes = set([gdal.GDT_Float32, gdal.GDT_Float64])


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
    
    if bandObj.DataType in gdalFloatTypes:
        # convert values away from numpy scalars as they have a repr()
        # in the form np.float... Band is float so conver to floats
        minVal = float(minVal)
        maxVal = float(maxVal)
        modeVal = float(modeVal)
        medianVal = float(medianVal)
    else:
        # convert to ints
        minVal = int(minVal)
        maxVal = int(maxVal)
        modeVal = int(modeVal)
        medianVal = int(medianVal)
    # mean and standard deviation stay as floats
    
    bandObj.SetMetadataItem("STATISTICS_MINIMUM", repr(minVal))
    bandObj.SetMetadataItem("STATISTICS_MAXIMUM", repr(maxVal))
    bandObj.SetMetadataItem("STATISTICS_MEAN", repr(float(meanVal)))
    bandObj.SetMetadataItem("STATISTICS_STDDEV", repr(float(stdDevVal)))
    bandObj.SetMetadataItem("STATISTICS_MODE", repr(modeVal))
    bandObj.SetMetadataItem("STATISTICS_MEDIAN", repr(medianVal))
    bandObj.SetMetadataItem("STATISTICS_SKIPFACTORX", "1")
    bandObj.SetMetadataItem("STATISTICS_SKIPFACTORY", "1")
    bandObj.SetMetadataItem("STATISTICS_HISTOBINFUNCTION", "direct")


def addOverviews(ds):
    """
    Add raster overviews to the given file.
    Mimic rios.calcstats behaviour to decide how many overviews.

    Parameters
    ----------
      ds : gdal.Dataset
        Open Dataset for the raster file

    """
    # first we work out how many overviews to build based on the size
    if ds.RasterXSize < ds.RasterYSize:
        mindim = ds.RasterXSize
    else:
        mindim = ds.RasterYSize
    
    nOverviews = 0
    for i in DEFAULT_OVERVIEWLEVELS:
        if (mindim // i) > DEFAULT_MINOVERVIEWDIM:
            nOverviews = nOverviews + 1
            
    ds.BuildOverviews("NEAREST", DEFAULT_OVERVIEWLEVELS[:nOverviews])


def writeRandomColourTable(outBand, nRows):
    """
    Attach a randomly-generated colour table to the given segmentation
    image. Mainly useful so the segmentation boundaries can be viewed,
    without any regard to the meaning of the segments.

    Parameters
    ----------
      outBand : gdal.Band
        Open GDAL Band object for the segmentation image
      nRows : int
        Number of rows in the attribute table, equal to the
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
    colNum = attrTbl.GetColOfUsage(gdal.GFU_Alpha)
    if colNum == -1:
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

    The general idea is that the given columns would be the per-segment 
    mean values of the desired bands (see tiling.calcPerSegmentStatsTiled()
    to create such columns). 

    Parameters
    ----------
      segfile : str or gdal.Dataset
        Filename of the completed segmentation image, with RAT columns
        already written.  Can be either the file name string, or
        an open Dataset object.
      redColName : str
        Name of the column in the RAT to use for the red color
      greenColName : str
        Name of the column in the RAT to use for the green color
      blueColName : str
        Name of the column in the RAT to use for the blue color

    """
    colList = [redColName, greenColName, blueColName]
    colorColList = ['Red', 'Green', 'Blue']
    usageList = [gdal.GFU_Red, gdal.GFU_Green, gdal.GFU_Blue]

    if isinstance(segfile, gdal.Dataset):
        ds = segfile
    else:
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


deprecationAlreadyWarned = set()


def deprecationWarning(msg, stacklevel=2):
    """
    Print a deprecation warning to stderr. Includes the filename
    and line number of the call to the function which called this.
    The stacklevel argument controls how many stack levels above this
    gives the line number.

    Implemented in mimcry of warnings.warn(), which seems very flaky.
    Sometimes it prints, and sometimes not, unless PYTHONWARNINGS is set
    (or -W is used). This function at least seems to work consistently.

    """
    frame = inspect.currentframe()
    for i in range(stacklevel):
        if frame is not None:
            frame = frame.f_back

    if frame is None:
        filename = "sys"
        lineno = 1
    else:
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno

    key = (filename, lineno)
    if key not in deprecationAlreadyWarned:
        print("{} (line {}):\n    WARNING: {}".format(filename, lineno, msg),
            file=sys.stderr)
        deprecationAlreadyWarned.add(key)


class WorkerErrorRecord:
    """
    Hold a record of an exception raised in a remote worker.
    """
    def __init__(self, exc, workerType):
        self.exc = exc
        self.workerType = workerType
        self.formattedTraceback = traceback.format_exception(exc)

    def __str__(self):
        headLine = "Error in {} worker".format(self.workerType)
        lines = [headLine]
        lines.extend([line.strip('\n') for line in self.formattedTraceback])
        s = '\n'.join(lines) + '\n'
        return s


def reportWorkerException(exceptionRecord):
    """
    Report the given WorkerExceptionRecord object to stderr
    """
    print(exceptionRecord, file=sys.stderr)


def formatTimingRpt(summaryDict):
    """
    Format a report on timings, given the output of Timers.makeSummaryDict()

    Return a single string of the formatted report.
    """
    isSeg = ('spectralclusters' in summaryDict)
    isStats = ('statscompletion' in summaryDict)
    if isSeg:
        hdr = "Segmentation Timings (sec)"
        timerList = ['spectralclusters', 'reading', 'segmentation',
            'stitchtiles']
    elif isStats:
        hdr = "Per-segment Stats Timings (sec)"
        timerList = ['reading', 'accumulation', 'statscompletion', 'writing']
    else:
        # Some unknown set of timers, do something sensible
        hdr = "Timers (unknown set) (sec)"
        timerList = sorted(list(summaryDict.keys()))

    lines = [hdr]
    walltimeDict = summaryDict.get('walltime')
    if walltimeDict is not None:
        walltime = walltimeDict['total']
        lines.append(f"Walltime: {walltime:.2f}")
    lines.append("")

    # Work out column widths and format strings. Very tedious, but neater output.
    fldWidth1 = max([len(t) for t in timerList])
    maxTime = max([summaryDict[t]['total'] for t in timerList])
    logMaxTime = numpy.log10(maxTime)
    if int(logMaxTime) == logMaxTime:
        # maxTime is exact power of 10, so force ceil() to go up anyway
        logMaxTime += 0.1
    fldWidth2 = 3 + int(numpy.ceil(logMaxTime))
    colHdrFmt = "{:" + str(fldWidth1) + "s}   {:>" + str(fldWidth2) + "s}"
    lines.append(colHdrFmt.format("Timer", "Total"))
    lines.append((3 + fldWidth1 + fldWidth2) * '-')
    colFmt = "{:" + str(fldWidth1) + "s}   {:" + str(fldWidth2) + ".2f}"

    # Now add the table of timings.
    for t in timerList:
        line = colFmt.format(t, summaryDict[t]['total'])
        lines.append(line)

    outStr = '\n'.join(lines)
    return outStr
