"""
Routines to support calculation of statistics on large rasters.
The statistics are calculated per segment so require two input
rasters - the segmentation output, and another image to gather 
statistics from, for each segment in the first image. 

These are optimised to work on one tile of the images at a
time so should be efficient in terms of memory use.

The main functions in this module are :func:`calcPerSegmentStatsTiled`
and :func:`calcPerSegmentSpatialStatsTiled`.

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

import sys
import numpy
from osgeo import gdal
from osgeo import osr

from numba import njit, typeof
from numba.core import types
from numba.typed import Dict, List
from numba.experimental import jitclass

from . import tiling
from . import shepseg


def calcPerSegmentStatsTiled(imgfile, imgbandnum, segfile, 
            statsSelection, missingStatsValue=-9999):
    """
    Calculate selected per-segment statistics for the given band 
    of the imgfile, against the given segment raster file. 
    Calculated statistics are written to the segfile raster 
    attribute table (RAT), so this file format must support RATs. 
    
    Calculations are carried out in a memory-efficient way, allowing 
    very large rasters to be processed. Raster data is handled in 
    small tiles, attribute table is handled in fixed-size chunks. 
    
    The statsSelection parameter is a list of tuples, one for each
    statistics to be included. Each tuple is either 2 or 3 elements,

        (columnName, statName) or (columnName, statName, parameter)

    The 3-element form is used for any statistic which requires
    a parameter, which currently is only the percentile. 
    
    The columnName is a string, used to name the column in the 
    output RAT. 
    The statName is a string used to identify which statistic 
    is to be calculated. Available options are:

        'min', 'max', 'mean', 'stddev', 'median', 'mode', 
        'percentile', 'pixcount'.

    The 'percentile' statistic requires the 3-element form, with 
    the 3rd element being the percentile to be calculated. 
    
    For example
        [('Band1_Mean', 'mean'),
         ('Band1_stdDev', 'stddev'),
         ('Band1_LQ', 'percentile', 25),
         ('Band1_UQ', 'percentile', 75)
        ]
    would create 4 columns, for the per-segment mean and 
    standard deviation of the given band, and the lower and upper 
    quartiles, with corresponding column names. 
    
    Any pixels that are set to the nodata value of imgfile (if set)
    are ignored in the stats calculations. If there are no pixels
    that aren't the nodata value then the value passed in as
    missingStatsValue is put into the RAT for the requested
    statistics.
    
    The 'pixcount' statName can be used to find the number of
    valid pixels (not nodata) that were used to calculate the statistics.

    """
    segds = segfile
    if not isinstance(segds, gdal.Dataset):
        segds = gdal.Open(segfile, gdal.GA_Update)
    segband = segds.GetRasterBand(1)

    imgds = imgfile
    if not isinstance(imgds, gdal.Dataset):
        imgds = gdal.Open(imgfile, gdal.GA_ReadOnly)
    imgband = imgds.GetRasterBand(imgbandnum)
    if (imgband.DataType == gdal.GDT_Float32 or 
            imgband.DataType == gdal.GDT_Float64):
        raise PyShepSegStatsError("Float image types not supported")
        
    if segband.XSize != imgband.XSize or segband.YSize != imgband.YSize:
        raise PyShepSegStatsError("Images must be same size")
        
    if segds.GetGeoTransform() != imgds.GetGeoTransform():
        raise PyShepSegStatsError("Images must have same spatial extent and pixel size")
        
    if not equalProjection(segds.GetProjection(), imgds.GetProjection()):
        raise PyShepSegStatsError("Images must be in the same projection")
    
    attrTbl = segband.GetDefaultRAT()
    existingColNames = [attrTbl.GetNameOfCol(i) 
        for i in range(attrTbl.GetColumnCount())]
        
    # Note: may be None if no value set
    imgNullVal = imgband.GetNoDataValue()
    if imgNullVal is not None:
        # cast to the same type we are using for imagery
        # (GDAL records this value as double)
        imgNullVal = tiling.numbaTypeForImageType(imgNullVal)
        
    histColNdx = checkHistColumn(existingColNames)
    segSize = attrTbl.ReadAsArray(histColNdx).astype(numpy.uint32)
    
    # Create columns, as required
    colIndexList = createStatColumns(statsSelection, attrTbl, existingColNames)
    (statsSelection_fast, numIntCols, numFloatCols) = (
        makeFastStatsSelection(colIndexList, statsSelection))

    # Loop over all tiles in image
    tileSize = tiling.TILESIZE
    (nlines, npix) = (segband.YSize, segband.XSize)
    numXtiles = int(numpy.ceil(npix / tileSize))
    numYtiles = int(numpy.ceil(nlines / tileSize))
    
    segDict = createSegDict()
    pagedRat = tiling.createPagedRat()
    noDataDict = createNoDataDict()
    
    for tileRow in range(numYtiles):
        for tileCol in range(numXtiles):
            topLine = tileRow * tileSize
            leftPix = tileCol * tileSize
            xsize = min(tileSize, npix - leftPix)
            ysize = min(tileSize, nlines - topLine)
            
            tileSegments = segband.ReadAsArray(leftPix, topLine, xsize, ysize)
            tileImageData = imgband.ReadAsArray(leftPix, topLine, xsize, ysize)
            
            accumulateSegDict(segDict, noDataDict, imgNullVal, tileSegments, 
                tileImageData)
            calcStatsForCompletedSegs(segDict, noDataDict, missingStatsValue, 
                pagedRat, statsSelection_fast, segSize, numIntCols, numFloatCols)
            
            writeCompletePages(pagedRat, attrTbl, statsSelection_fast)

    # all pages should now be written. Raise an error if this not the case.
    if len(pagedRat) > 0:
        raise PyShepSegStatsError('Not all pixels found during processing')


@njit
def accumulateSegDict(segDict, noDataDict, imgNullVal, tileSegments, tileImageData):
    """
    Accumulate per-segment histogram counts for all 
    pixels in the given tile. Updates segDict entries in-place. 
    """
    ysize, xsize = tileSegments.shape
    
    for y in range(ysize):
        for x in range(xsize):
            segId = tileSegments[y, x]
            if segId != shepseg.SEGNULLVAL:
            
                # always create a empty dictionary for this segId
                # even if we haven't got any non-nodata pixels yet
                # because we loop through keys of segDict when calculating stats
                if segId not in segDict:
                    segDict[segId] = Dict.empty(key_type=tiling.numbaTypeForImageType, 
                        value_type=types.uint32)
                
                imgVal = tileImageData[y, x]
                imgVal_typed = tiling.numbaTypeForImageType(imgVal)
                if imgNullVal is not None and imgVal_typed == imgNullVal:
                    # this is the null value for the tileImageData
                    if segId not in noDataDict:
                        noDataDict[segId] = types.uint32(0)
                        
                    noDataDict[segId] = types.uint32(noDataDict[segId] + 1)

                else:
                    # else populate histogram (note: not done if all nodata for this segment)
                    d = segDict[segId]
                    if imgVal_typed not in d:
                        d[imgVal_typed] = types.uint32(0)

                    d[imgVal_typed] = types.uint32(d[imgVal_typed] + 1)


@njit
def checkSegComplete(segDict, noDataDict, segSize, segId):
    """
    Return True if the given segment has a complete entry
    in the segDict, meaning that the pixel count is equal to
    the segment size
    """
    count = 0
    # add up the counts of the histogram
    if segId in segDict:
        d = segDict[segId]
        for pixVal in d:
            count += d[pixVal]
        
    # now add up any nodata in this segment
    if segId in noDataDict:
        count += noDataDict[segId]
        
    return (count == segSize[segId])


@njit
def calcStatsForCompletedSegs(segDict, noDataDict, missingStatsValue, pagedRat, 
        statsSelection_fast, segSize, numIntCols, numFloatCols):
    """
    Calculate statistics for all complete segments in the segDict.
    Update the pagedRat with the resulting entries. Completed segments
    are then removed from segDict. 
    """
    numStats = len(statsSelection_fast)
    maxSegId = len(segSize) - 1
    segDictKeys = numpy.empty(len(segDict), dtype=tiling.segIdNumbaType)
    i = 0
    for segId in segDict:
        segDictKeys[i] = segId
        i += 1
    for segId in segDictKeys:
        segComplete = checkSegComplete(segDict, noDataDict, segSize, segId)
        if segComplete:
            segStats = SegmentStats(segDict[segId], missingStatsValue)
            ratPageId = tiling.getRatPageId(segId)
            if ratPageId not in pagedRat:
                numSegThisPage = min(tiling.RAT_PAGE_SIZE, 
                    (maxSegId - ratPageId + 1))
                pagedRat[ratPageId] = tiling.RatPage(numIntCols, numFloatCols, 
                    ratPageId, numSegThisPage)
            ratPage = pagedRat[ratPageId]
            for i in range(numStats):
                statId = statsSelection_fast[i, STATSEL_STATID]
                param = statsSelection_fast[i, STATSEL_PARAM]
                val = segStats.getStat(statId, param)
                
                colType = statsSelection_fast[i, STATSEL_COLTYPE]
                colArrayNdx = statsSelection_fast[i, STATSEL_COLARRAYINDEX]
                ratPage.setRatVal(segId, colType, colArrayNdx, val)

            ratPage.setSegmentComplete(segId)
            
            # Stats now done for this segment, so remove its histogram
            segDict.pop(segId)
            # same for nodata (if there is one)
            if segId in noDataDict:
                noDataDict.pop(segId)


def createSegDict():
    """
    Create the Dict of Dicts for handling per-segment histograms. 
    Each entry is a dictionary, and the key is a segment ID.
    Each dictionary within this is the per-segment histogram for
    a single segment. Each of its entries is for a single value from 
    the imagery, the key is the pixel value, and the dictionary value 
    is the number of times that pixel value appears in the segment. 
    """
    histDict = Dict.empty(key_type=tiling.numbaTypeForImageType, value_type=types.uint32)
    segDict = Dict.empty(key_type=tiling.segIdNumbaType, value_type=histDict._dict_type)
    return segDict


def createNoDataDict():
    """
    Create the dictionary that holds counts of nodata seen for each 
    segment. The key is the segId, value is the count of nodata seen 
    for that segment in the image data.
    """
    noDataDict = Dict.empty(key_type=tiling.segIdNumbaType, value_type=types.uint32)
    return noDataDict


def checkHistColumn(existingColNames):
    """
    Check for the Histogram column in the attribute table. Return
    its column number, and raise an exception if it is not present
    """
    histColNdx = -1
    for i in range(len(existingColNames)):
        if existingColNames[i] == 'Histogram':
            histColNdx = i
    if histColNdx < 0:
        msg = "Histogram column must exist before calculating per-segment stats"
        raise PyShepSegStatsError(msg)
    return histColNdx


def createStatColumns(statsSelection, attrTbl, existingColNames):
    """
    Create requested statistic columns on the segmentation image RAT.
    Statistic columns are of type gdal.GFT_Real for mean and stddev, 
    and gdal.GFT_Integer for all other statistics. 
    
    Return the column indexes for all requested columns, in the same
    order. 
    """
    colIndexList = []
    for selection in statsSelection:
        (colName, statName) = selection[:2]
        if colName not in existingColNames:
            colType = gdal.GFT_Integer
            if statName in ('mean', 'stddev'):
                colType = gdal.GFT_Real
            attrTbl.CreateColumn(colName, colType, gdal.GFU_Generic)
            colNdx = attrTbl.GetColumnCount() - 1
        else:
            print('Column {} already exists'.format(colName))
            colNdx = existingColNames.index(colName)
        colIndexList.append(colNdx)
    return colIndexList


def writeCompletePages(pagedRat, attrTbl, statsSelection_fast):
    """
    Check for completed pages, and write them to the attribute table.
    Remove them from the pagedRat after writing. 
    """
    numStat = len(statsSelection_fast)
    
    pagedRatKeys = numpy.empty(len(pagedRat), dtype=shepseg.SegIdType)
    i = 0
    for pageId in pagedRat:
        pagedRatKeys[i] = pageId
        i += 1

    for pageId in pagedRatKeys:
        ratPage = pagedRat[pageId]
        if ratPage.pageComplete():
            startSegId = ratPage.startSegId
            for i in range(numStat):
                statSel = statsSelection_fast[i]
                colNumber = int(statSel[STATSEL_GLOBALCOLINDEX])
                colType = statSel[STATSEL_COLTYPE]
                colArrayNdx = statSel[STATSEL_COLARRAYINDEX]
                if colType == tiling.STAT_DTYPE_INT:
                    colArr = ratPage.intcols[colArrayNdx]
                elif colType == tiling.STAT_DTYPE_FLOAT:
                    colArr = ratPage.floatcols[colArrayNdx]

                attrTbl.WriteArray(colArr, colNumber, start=startSegId)
            
            # Remove page after writing. 
            pagedRat.pop(pageId)


# Translate statistic name strings into integer ID values
STATID_MIN = 0
STATID_MAX = 1
STATID_MEAN = 2
STATID_STDDEV = 3
STATID_MEDIAN = 4
STATID_MODE = 5
STATID_PERCENTILE = 6
STATID_PIXCOUNT = 7
statIDdict = {
    'min': STATID_MIN,
    'max': STATID_MAX,
    'mean': STATID_MEAN,
    'stddev': STATID_STDDEV,
    'median': STATID_MEDIAN,
    'mode': STATID_MODE,
    'percentile': STATID_PERCENTILE,
    'pixcount': STATID_PIXCOUNT
}
NOPARAM = -1

# Array indexes for the fast stat selection array
STATSEL_GLOBALCOLINDEX = 0
STATSEL_STATID = 1
STATSEL_COLTYPE = 2
STATSEL_COLARRAYINDEX = 3
STATSEL_PARAM = 4


def makeFastStatsSelection(colIndexList, statsSelection):
    """
    Make a fast version of the statsSelection data structure, combined
    with the global column index numbers.
    
    Return a tuple of 
        (statsSelection_fast, numIntCols, numFloatCols)
    The statsSelection_fast is a single array, of shape (numStats, 5). 
    The first index corresponds to the sequence in statsSelection. 
    The second index corresponds to the STATSEL_* values. 
    
    Everything is encoded as an integer value in a single numpy array, 
    suitable for fast access within numba njit-ed functions. 
    
    This is all a bit ugly and un-pythonic. Not sure if there is
    a better way. 
    
    """
    numStats = len(colIndexList)
    statsSelection_fast = numpy.empty((numStats, 5), dtype=numpy.uint32)
    
    intCount = 0
    floatCount = 0
    for i in range(numStats):
        statsSelection_fast[i, STATSEL_GLOBALCOLINDEX] = colIndexList[i]
        
        statName = statsSelection[i][1]
        statId = statIDdict[statName]
        statsSelection_fast[i, STATSEL_STATID] = statId
        
        statType = tiling.STAT_DTYPE_INT
        if statName in ('mean', 'stddev'):
            statType = tiling.STAT_DTYPE_FLOAT
        statsSelection_fast[i, STATSEL_COLTYPE] = statType
        
        if statType == tiling.STAT_DTYPE_INT:
            statsSelection_fast[i, STATSEL_COLARRAYINDEX] = intCount
            intCount += 1
        elif statType == tiling.STAT_DTYPE_FLOAT:
            statsSelection_fast[i, STATSEL_COLARRAYINDEX] = floatCount
            floatCount += 1

        statsSelection_fast[i, STATSEL_PARAM] = NOPARAM
        if statName == 'percentile':
            statsSelection_fast[i, STATSEL_PARAM] = statsSelection[i][2]
    
    return (statsSelection_fast, intCount, floatCount)


@njit
def getSortedKeysAndValuesForDict(d):
    """
    The given dictionary is keyed by pixel values from the imagery,
    and the values are counts of occurences of the corresponding pixel
    value. This function returns a pair of numpy arrays (as a tuple),
    one for the list of pixel values, and one for the corresponding
    counts. The arrays are sorted in increasing order of pixel value.
    """
    size = len(d)
    keysArray = numpy.empty(size, dtype=tiling.numbaTypeForImageType)
    valuesArray = numpy.empty(size, dtype=numpy.uint32)
    
    dictKeys = d.keys()
    c = 0
    for key in dictKeys:
        keysArray[c] = key
        valuesArray[c] = d[key]
        c += 1
    
    index = numpy.argsort(keysArray)
    keysSorted = keysArray[index]
    valuesSorted = valuesArray[index]
    
    return keysSorted, valuesSorted


# Warning - currently using uint32 or float32 for all of the types
# which should really be dependent on the imagery datatype. 
# Not sure whether it is possible to do better. 
segStatsSpec = [('pixVals', tiling.numbaTypeForImageType[:]), 
                ('counts', types.uint32[:]),
                ('pixCount', types.uint32),
                ('min', tiling.numbaTypeForImageType),
                ('max', tiling.numbaTypeForImageType),
                ('mean', types.float32),
                ('stddev', types.float32),
                ('median', tiling.numbaTypeForImageType),
                ('mode', tiling.numbaTypeForImageType),
                ('missingStatsValue', tiling.numbaTypeForImageType)
                ]


@jitclass(segStatsSpec)
class SegmentStats(object):
    "Manage statistics for a single segment"
    def __init__(self, segmentHistDict, missingStatsValue):
        """
        Construct with generic statistics, given a typed 
        dictionary of the histogram counts of all values
        in the segment.
        
        If there are no valid pixels then the value passed
        in as missingStatsValue is returned for the requested
        stats.
        
        """
        self.pixVals, self.counts = getSortedKeysAndValuesForDict(segmentHistDict)
        
        # Total number of pixels in segment
        self.pixCount = self.counts.sum()
        
        self.missingStatsValue = missingStatsValue
        
        if self.pixCount == 0:
            # all nodata
            self.min = missingStatsValue
            self.max = missingStatsValue
            self.mean = missingStatsValue
            self.stddev = missingStatsValue
            self.mode = missingStatsValue
            self.median = missingStatsValue
        else:
            # Min and max pixel values
            self.min = self.pixVals[0]
            self.max = self.pixVals[-1]

            # Mean value
            self.mean = (self.pixVals * self.counts).sum() / self.pixCount

            # Standard deviation
            variance = (self.counts * (self.pixVals - self.mean)**2).sum() / self.pixCount
            self.stddev = numpy.sqrt(variance)

            # Mode
            self.mode = self.pixVals[numpy.argmax(self.counts)]

            # Median
            self.median = self.getPercentile(50)
        
    def getPercentile(self, percentile):
        """
        Return the pixel value for the given percentile, 
        e.g. getPercentile(50) would return the median value of 
        the segment
        """
        if self.pixCount == 0:
            # all nodata
            return self.missingStatsValue
        else:
            countAtPcntile = self.pixCount * (percentile / 100)
            cumCount = 0
            i = 0
            while cumCount < countAtPcntile:
                cumCount += self.counts[i]
                i += 1
            pcntileVal = self.pixVals[i - 1]
            return pcntileVal
    
    def getStat(self, statID, param):
        """
        Return the requested statistic
        """
        if statID == STATID_MIN:
            val = self.min
        elif statID == STATID_MAX:
            val = self.max
        elif statID == STATID_MEAN:
            val = self.mean
        elif statID == STATID_STDDEV:
            val = self.stddev
        elif statID == STATID_MEDIAN:
            val = self.median
        elif statID == STATID_MODE:
            val = self.mode
        elif statID == STATID_PERCENTILE:
            val = self.getPercentile(param)
        elif statID == STATID_PIXCOUNT:
            val = self.pixCount
        return val


def equalProjection(proj1, proj2):
    """
    Returns True if the proj1 is the same as proj2
    
    Stolen from rios/pixelgrid.py

    """
    selfProj = str(proj1) if proj1 is not None else ''
    otherProj = str(proj2) if proj2 is not None else ''
    srSelf = osr.SpatialReference(wkt=selfProj)
    srOther = osr.SpatialReference(wkt=otherProj)
    return bool(srSelf.IsSame(srOther))


@njit
def userFuncVariogram(pts, imgNullVal, intArr, floatArr, maxDist):
    """
    Calculates the variogram at the given distance for the segment
    contained in the tile. This function is intended to be passed in as the 
    ``userFunc`` parameter to :func:`calcPerSegmentSpatialStatsTiled` if 
    variograms are to be calculated.
    
    ``maxDist`` is the number of variograms to calculate and should be 
    passed in as the ``userParam`` argument to 
    :func:`calcPerSegmentSpatialStatsTiled`.
    
    It is assumed that floatArr has enough space for the number of variograms
    requested (this is calculated from the ``colNamesAndTypes`` parameter to
    :func:`calcPerSegmentSpatialStatsTiled`).

    """
    # turn into a tile for maximum speed when searching
    tile = convertPtsInto2DArray(pts, imgNullVal)
    
    counts = numpy.zeros((maxDist,), dtype=numpy.uint32)
    sumDifSqs = numpy.zeros((maxDist,), dtype=numpy.float64)
    ysize, xsize = tile.shape
    for y in range(ysize):
        for x in range(xsize):
            val = tile[y, x]
            if val == imgNullVal:
                continue
                
            for yoffset in range(1, maxDist + 1):
                for xoffset in range(1, maxDist + 1):
                    if (y + yoffset) < ysize and (x + xoffset) < xsize:
                        val2 = tile[y + yoffset, x + xoffset]
                        if val2 == imgNullVal:
                            continue                       

                        # note: 'bin' dist by converting to integer
                        dist = int(numpy.sqrt(yoffset * yoffset + xoffset * xoffset))
                        if dist <= maxDist and dist > 0:
                            counts[dist - 1] += 1
                            sumDifSqs[dist - 1] += (val - val2)**2
                        
    for n in range(maxDist):
        if counts[n] > 0:
            floatArr[n] = numpy.sqrt(sumDifSqs[n] / counts[n])


@njit
def userFuncMeanCoord(pts, imgNullVal, intArr, floatArr, transform):
    """
    Calculates the mean coordinate of each segment. This function is intended 
    to be passed in as the ``userFunc`` parameter to 
    :func:`calcPerSegmentSpatialStatsTiled` if the mean coordinates of each
    segment are required. 

    The ``transform`` of the image (ie. returned by ds.GetGeoTransform() where
    ds is a GDAL dataset) is to be passed as ``userParam`` but converted to an
    array so it works with Numba.
    
    The result will be written to floatArr (2 values -
    easting then northing). It is expected that at least 2 float columns are
    available. 
    """
    count = 0
    sumx = 0.0
    sumy = 0.0
    for pt in pts:
    
        # can't use gdal.AppyGetTransform since we are within Numba.
        geox = transform[0] + transform[1] * pt.x + transform[2] * pt.y
        geoy = transform[3] + transform[4] * pt.x + transform[5] * pt.y
        
        sumx += geox
        sumy += geoy
        count += 1
        
    floatArr[0] = sumx / count
    floatArr[1] = sumy / count


SegPointSpec = [('x', types.uint32), 
    ('y', types.uint32), 
    ('val', tiling.numbaTypeForImageType)]
    

@jitclass(SegPointSpec)
class SegPoint(object):
    """
    Class for handling a given data point and it's location
    in pixel space (within the whole image, not a tile).
    
    Used so that all the data for a given segment can be collected
    together even if the segment straddles a tile.
    """
    def __init__(self, x, y, val):
        self.x = x
        self.y = y
        self.val = val

        
# for use in jitted functions - see https://github.com/numba/numba/issues/7291
if 'sphinx' not in sys.modules:
    # for some reason fails when called from sphinx
    PTS_TYPE = typeof(SegPoint(0, 0, 0))


def createSegSpatialDataDict():
    """
    Create a dictionary where the key is the segment ID and the 
    value is a List of :class:`SegPoint` objects.
    """
    pointList = List.empty_list(SegPoint.class_type.instance_type)
    segDict = Dict.empty(key_type=tiling.segIdNumbaType, 
                value_type=pointList._list_type)
    return segDict
    

def calcPerSegmentSpatialStatsTiled(imgfile, imgbandnum, segfile,
        colNamesAndTypes, userFunc, userParam=None, missingStatsValue=-9999):
    """
    Similar to the :func:`calcPerSegmentSpatialStatsTiled` function 
    but allows the user to calculate spatial statistics on the data
    for each segment. This is done by recording the location and value
    for each pixel within the segment. Once all the pixels are found
    for a segment the ``userFunc`` is called with the following parameters:
    
    ::
    
        pts, imgNullVal, intArr, floatArr, userParam
        
    where ``pts`` is List of :class:`SegPoint` objects. If 2D Numpy tile is prefered the
    ``userFunc`` can call :func:`convertPtsInto2DArray`. ``intArray`` is a 
    1D numpy array which all the integer output values are to be put (in the 
    same order given in ``colNamesAndTypes``). ``floatArr`` is a 1D numpy array 
    which all the floating point output values are to be put (in the same order
    given in ``colNamesAndTypes``). These arrays are initialised with 
    ``missingStatsValue`` so the function can skip any that it doesn't have
    values for. ``userParam`` is the same value passed to 
    this function and needs to be a type understood by Numba. 
    
    ``userFunc`` needs to be a Numba function (ie decorated with @jit or @njit).
    
    ``colNamesAndTypes`` should be a list of ``(colName, colType)`` tuples that
    define the names, types are order of the output RAT columns. 
    ``colName`` should be a string containing the name of the RAT column to be
    created and ``colType`` should be one of ``gdal.GFT_Integer`` or 
    ``gdal.GFT_Real`` and this controls the type of the column to be created.
    Note that the order of columns given in this parameter is important as
    this dicates the order of the ``intArray`` and ``floatArr`` parameters to 
    ``userFunc``.
    
    """
    segds = segfile
    if not isinstance(segds, gdal.Dataset):
        segds = gdal.Open(segfile, gdal.GA_Update)
    segband = segds.GetRasterBand(1)

    imgds = imgfile
    if not isinstance(imgds, gdal.Dataset):
        imgds = gdal.Open(imgfile, gdal.GA_ReadOnly)
    imgband = imgds.GetRasterBand(imgbandnum)
    if (imgband.DataType == gdal.GDT_Float32 or 
            imgband.DataType == gdal.GDT_Float64):
        raise PyShepSegStatsError("Float image types not supported")
        
    if segband.XSize != imgband.XSize or segband.YSize != imgband.YSize:
        raise PyShepSegStatsError("Images must be same size")
        
    if segds.GetGeoTransform() != imgds.GetGeoTransform():
        raise PyShepSegStatsError("Images must have same spatial extent and pixel size")
        
    if not equalProjection(segds.GetProjection(), imgds.GetProjection()):
        raise PyShepSegStatsError("Images must be in the same projection")
    
    attrTbl = segband.GetDefaultRAT()
    existingColNames = [attrTbl.GetNameOfCol(i) 
        for i in range(attrTbl.GetColumnCount())]
        
    # Note: may be None if no value set
    imgNullVal = imgband.GetNoDataValue()
    if imgNullVal is not None:
        # cast to the same type we are using for imagery
        # (GDAL records this value as double)
        imgNullVal = tiling.numbaTypeForImageType(imgNullVal)
    else:
        # because we need to mask out parts of tiles not part of the
        # segment we need the no data value set
        raise PyShepSegStatsError("NoData value must be set on imgfile")
        
    if 'targetoptions' not in userFunc.__dict__:
        raise PyShepSegStatsError("userFunc must be @jit or @njit decorated")
        
    if len(colNamesAndTypes) == 0:
        raise PyShepSegStatsError("Must specify one or more columns")
    
    histColNdx = checkHistColumn(existingColNames)
    segSize = attrTbl.ReadAsArray(histColNdx).astype(numpy.uint32)
    
    # Create columns, as required 
    n_intCols, n_floatCols, statsSelection_fast = createUserColumnsSpatial(
        colNamesAndTypes, attrTbl, existingColNames)
    # create temprorary arrays for userfunc
    intArr = numpy.empty(n_intCols, dtype=numpy.int32)
    floatArr = numpy.empty(n_floatCols, dtype=numpy.float64)
        
    # Loop over all tiles in image
    tileSize = tiling.TILESIZE
    (nlines, npix) = (segband.YSize, segband.XSize)
    numXtiles = int(numpy.ceil(npix / tileSize))
    numYtiles = int(numpy.ceil(nlines / tileSize))
    
    segDict = createSegSpatialDataDict()
    pagedRat = tiling.createPagedRat()
    noDataDict = createNoDataDict()
    
    # similar logic to calcPerSegmentSpatialStatsTiled
    for tileRow in range(numYtiles):
        for tileCol in range(numXtiles):
            topLine = tileRow * tileSize
            leftPix = tileCol * tileSize
            xsize = min(tileSize, npix - leftPix)
            ysize = min(tileSize, nlines - topLine)
            
            tileSegments = segband.ReadAsArray(leftPix, topLine, xsize, ysize)
            tileImageData = imgband.ReadAsArray(leftPix, topLine, xsize, ysize)
            
            accumulateSegSpatial(segDict, noDataDict, imgNullVal, tileSegments, 
                tileImageData, topLine, leftPix)
            calcStatsForCompletedSegsSpatial(segDict, noDataDict, 
                missingStatsValue, pagedRat, segSize, userFunc, userParam, 
                statsSelection_fast, intArr, floatArr, imgNullVal)
            
            writeCompletePages(pagedRat, attrTbl, statsSelection_fast)

    # all pages should now be written. Raise an error if this not the case.
    if len(pagedRat) > 0:
        raise PyShepSegStatsError('Not all pixels found during processing')
        

def createUserColumnsSpatial(colNamesAndTypes, attrTbl, existingColNames):
    """
    Used by :func:`calcPerSegmentSpatialStatsTiled` to create columns specified
    in the ``colNamesAndTypes`` structure. 
    
    Returns a tuple with number of integer columns, number of float columns
    and a statsSelection_fast array for use by :func:`calcStatsForCompletedSegsSpatial`.
    """
    n_intCols = 0
    n_floatCols = 0
    numStats = len(colNamesAndTypes)
    statsSelection_fast = numpy.empty((numStats, 5), dtype=numpy.uint32)
    
    for i, (colName, colType) in enumerate(colNamesAndTypes):
        if colName not in existingColNames:
            attrTbl.CreateColumn(colName, colType, gdal.GFU_Generic)
            colNdx = attrTbl.GetColumnCount() - 1
        else:
            colNdx = existingColNames.index(colName)
            if colType == attrTbl.GetTypeOfCol(colNdx):
                print('Column {} already exists'.format(colName))
            else:
                msg = 'Column {} already exists and is of differing type'.format(colName)
                raise PyShepSegStatsError(msg)

        statsSelection_fast[i, STATSEL_GLOBALCOLINDEX] = colNdx
        # not used
        statsSelection_fast[i, STATSEL_PARAM] = NOPARAM
        statsSelection_fast[i, STATSEL_STATID] = -1
                
        if colType == gdal.GFT_Integer:
            statsSelection_fast[i, STATSEL_COLARRAYINDEX] = n_intCols
            statsSelection_fast[i, STATSEL_COLTYPE] = tiling.STAT_DTYPE_INT
            n_intCols += 1
        elif colType == gdal.GFT_Real:
            statsSelection_fast[i, STATSEL_COLARRAYINDEX] = n_floatCols
            statsSelection_fast[i, STATSEL_COLTYPE] = tiling.STAT_DTYPE_FLOAT
            n_floatCols += 1
        else:
            msg = 'Unknown type ({}) for column {}'.format(colType, colName)
            raise PyShepSegStatsError(msg)

    return n_intCols + 1, n_floatCols + 1, statsSelection_fast


@njit
def accumulateSegSpatial(segDict, noDataDict, imgNullVal, tileSegments, 
                tileImageData, topLine, leftPix):
    """
    Accumulates data for each segment for the given tile. The data is put
    into segDict.
    
    """
    ysize, xsize = tileSegments.shape
    
    for y in range(ysize):
        for x in range(xsize):
            segId = tileSegments[y, x]
            if segId != shepseg.SEGNULLVAL:    
                if segId not in segDict:
                    segDict[segId] = List.empty_list(PTS_TYPE)
                    
                imgVal = tileImageData[y, x]
                imgVal_typed = tiling.numbaTypeForImageType(imgVal)
                if imgNullVal is not None and imgVal_typed == imgNullVal:
                    # this is the null value for the tileImageData
                    if segId not in noDataDict:
                        noDataDict[segId] = types.uint32(0)
                        
                    noDataDict[segId] = types.uint32(noDataDict[segId] + 1)

                else:
                    # else populate list (note: not done if all nodata for this segment)
                    segList = segDict[segId]
                    pt = SegPoint(leftPix + x, topLine + y, imgVal_typed)
                    segList.append(pt)


@njit
def checkSegCompleteSpatial(segDict, noDataDict, segSize, segId):
    """
    Return True if the given segment has a complete entry
    in the segDict, meaning that the pixel count is equal to
    the segment size
    """
    count = 0
    # add up the counts of the histogram
    if segId in segDict:
        segList = segDict[segId]
        count += len(segList)
        
    # now add up any nodata in this segment
    if segId in noDataDict:
        count += noDataDict[segId]
        
    return (count == segSize[segId])

    
@njit
def convertPtsInto2DArray(pts, imgNullVal):
    """
    Given a list of points for a segment turn this back into a 2D array
    for passing to the user function. 
    
    The tile is created just large enough for the shape of the segment. 
    Areas of the tile not within a segment is given the value of ``imgNullVal``.
    
    """
    # find size of tile
    xmin = pts[0].x
    xmax = pts[0].x
    ymin = pts[0].y
    ymax = pts[0].y
    for p in pts[1:]:
        if p.x < xmin:
            xmin = p.x
        elif p.x > xmax:
            xmax = p.x
        if p.y < ymin:
            ymin = p.y
        elif p.y > ymax:
            ymax = p.y

    # create a new 2D array just big enough to hold the data and set to
    # imgNullVal to start with.
    xsize = xmax - xmin
    ysize = ymax - ymin
    tile = numpy.full((ysize + 1, xsize + 1), imgNullVal, 
            dtype=tiling.numbaTypeForImageType)
            
    # fill in the tile with the values
    for p in pts:
        tile[p.y - ymin, p.x - xmin] = p.val
        
    return tile


@njit
def calcStatsForCompletedSegsSpatial(segDict, noDataDict, missingStatsValue, 
        pagedRat, segSize, userFunc, userParam, statsSelection_fast, intArr, 
        floatArr, imgNullVal):
    """
    Calls the ``userFunc`` on data for completed segs and saves the results into
    the ``pagedRat``. 
    
    """

    maxSegId = len(segSize) - 1
    segDictKeys = numpy.empty(len(segDict), dtype=tiling.segIdNumbaType)
    i = 0
    for segId in segDict:
        segDictKeys[i] = segId
        i += 1
    for segId in segDictKeys:
        segComplete = checkSegCompleteSpatial(segDict, noDataDict, segSize, segId)
        if segComplete:
            ratPageId = tiling.getRatPageId(segId)
            if ratPageId not in pagedRat:
                numSegThisPage = min(tiling.RAT_PAGE_SIZE, (maxSegId - ratPageId + 1))
                pagedRat[ratPageId] = tiling.RatPage(intArr.shape[0], floatArr.shape[0], 
                    ratPageId, numSegThisPage)
            ratPage = pagedRat[ratPageId]                   

            segList = segDict[segId]
            if len(segList) > 0:
                # initialise the arrays to missingStatsValue
                intArr.fill(missingStatsValue)
                floatArr.fill(missingStatsValue)
                
                # call userFunc
                userFunc(segList, imgNullVal, intArr, floatArr, userParam)
                
                # now write the outputs for the different types 
                for n in range(statsSelection_fast.shape[0]):
                    colType = statsSelection_fast[n, STATSEL_COLTYPE]
                    colArrayNdx = statsSelection_fast[n, STATSEL_COLARRAYINDEX]
                    if colType == tiling.STAT_DTYPE_INT:
                        ratPage.setRatVal(segId, tiling.STAT_DTYPE_INT, colArrayNdx,
                            intArr[colArrayNdx])
                    else:
                        ratPage.setRatVal(segId, tiling.STAT_DTYPE_FLOAT, colArrayNdx,
                            floatArr[colArrayNdx])
 
            else:
                # all missingStatsValue
                for n in range(statsSelection_fast.shape[0]):
                    colType = statsSelection_fast[n, STATSEL_COLTYPE]
                    colArrayNdx = statsSelection_fast[n, STATSEL_COLARRAYINDEX]
                    ratPage.setRatVal(segId, colType, colArrayNdx, missingStatsValue)
            # mark as complete
            ratPage.setSegmentComplete(segId)
            
            # Stats now done for this segment, so remove its points
            segDict.pop(segId)
            # same for nodata (if there is one)
            if segId in noDataDict:
                noDataDict.pop(segId)
                

class PyShepSegStatsError(Exception):
    pass
