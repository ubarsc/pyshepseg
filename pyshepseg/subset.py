"""
This module contains functionality for subsetting a large
segmented image into a smaller one for checking etc. See
:func:`subsetImage`.

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

import numpy
from numba import njit
from numba.typed import Dict
from osgeo import gdal

from . import shepseg
from . import tiling


def subsetImage(inname, outname, tlx, tly, newXsize, newYsize, outformat,
        creationOptions=[], origSegIdColName=None, maskImage=None):
    """
    Subset an image and "compress" the RAT so only values that
    are in the new image are in the RAT. Note that the image values
    will be recoded in the process.

    gdal_translate seems to have a problem with files that
    have large RAT's so while that gets fixed do the subsetting
    in this function.

    Parameters
    ----------
      inname : str
        Filename of input raster
      outname : str
        Filename of output raster
      tlx, tly : int
        The x & y pixel coordinates (i.e. col & row, respectively) of the
        top left pixel of the image subset to extract.
      newXsize, newYsize : int
        The x & y size (in pixels) of the image subset to extract
      outformat : str
        The GDAL short name of the format driver to use for the output
        raster file
      creationOptions : list of str
        The GDAL creation options for the output file
      origSegIdColName : str or None
        The name of a RAT column. If not None, this will be created
        in the output file, and will contain the original segment ID
        numbers so the new segment IDs can be linked back to the old
      maskImage : str or None
        If not None, then the filename of a mask raster. Only pixels
        which are non-zero in this mask image will be included in the
        subset. This image is assumed to match the shape and position
        of the output subset

    """
    inds = gdal.Open(inname)
    inband = inds.GetRasterBand(1)

    if (tlx + newXsize) > inband.XSize or (tly + newYsize) > inband.YSize:
        msg = 'Requested subset is not within input image'
        raise PyShepSegSubsetError(msg)

    driver = gdal.GetDriverByName(outformat)
    outds = driver.Create(outname, newXsize, newYsize, 1, inband.DataType,
                options=creationOptions)
    # set the output projection and transform
    outds.SetProjection(inds.GetProjection())
    transform = list(inds.GetGeoTransform())
    transform[0] = transform[0] + transform[1] * tlx
    transform[3] = transform[3] + transform[5] * tly
    outds.SetGeoTransform(transform)

    outband = outds.GetRasterBand(1)
    outband.SetMetadataItem('LAYER_TYPE', 'thematic')
    outRAT = outband.GetDefaultRAT()

    inRAT = inband.GetDefaultRAT()
    recodeDict = Dict.empty(key_type=tiling.segIdNumbaType,
        value_type=tiling.segIdNumbaType)  # keyed on original ID - value is new row ID
    histogramDict = Dict.empty(key_type=tiling.segIdNumbaType,
        value_type=tiling.segIdNumbaType)  # keyed on new ID - value is count

    # make the output file has the same columns as the input
    numIntCols, numFloatCols = copyColumns(inRAT, outRAT)

    # If a maskImage was specified then open it
    maskds = None
    maskBand = None
    maskData = None
    if maskImage is not None:
        maskds = gdal.Open(maskImage)
        maskBand = maskds.GetRasterBand(1)
        if maskBand.XSize != newXsize or maskBand.YSize != newYsize:
            msg = 'mask should match requested subset size if supplied'
            raise PyShepSegSubsetError(msg)

    # work out how many tiles we have
    tileSize = tiling.TILESIZE
    numXtiles = int(numpy.ceil(newXsize / tileSize))
    numYtiles = int(numpy.ceil(newYsize / tileSize))

    minInVal = None
    maxInVal = None

    for tileRow in range(numYtiles):
        for tileCol in range(numXtiles):
            leftPix = tlx + tileCol * tileSize
            topLine = tly + tileRow * tileSize
            xsize = min(tileSize, newXsize - leftPix + tlx)
            ysize = min(tileSize, newYsize - topLine + tly)

            # extract the image data for this tile from the input file
            inData = inband.ReadAsArray(leftPix, topLine, xsize, ysize)

            # work out the range of data for accessing the whole RAT (below)
            inDataMasked = inData[inData != shepseg.SEGNULLVAL]
            if len(inDataMasked) == 0:
                # no actual data in this tile
                continue

            minVal = inDataMasked.min()
            maxVal = inDataMasked.max()
            if minInVal is None or minVal < minInVal:
                minInVal = minVal
            if maxInVal is None or maxVal > maxInVal:
                maxInVal = maxVal

            if maskBand is not None:
                # if a mask file was specified read the corresoponding data
                maskData = maskBand.ReadAsArray(tileCol * tileSize,
                                tileRow * tileSize, xsize, ysize)

            # process this tile, obtaining the image of the 'new' segment ids
            # and updating recodeDict as we go
            outData = processSubsetTile(inData, recodeDict,
                        histogramDict, maskData)

            # write out the new segment ids to the output
            outband.WriteArray(outData, tileCol * tileSize, tileRow * tileSize)

    if minInVal is None or maxInVal is None:
        # must be all shepseg.SEGNULLVAL
        raise PyShepSegSubsetError('No valid data found in subset')

    # process the recodeDict, one page of the input at a time

    # fill this in as we go and write out each page when complete.
    outPagedRat = tiling.createPagedRat()
    for startSegId in range(minInVal, maxInVal, tiling.RAT_PAGE_SIZE):
        # looping through in tiling.RAT_PAGE_SIZE pages
        endSegId = min(startSegId + tiling.RAT_PAGE_SIZE - 1, maxInVal)

        # get this page in
        inPage = readRATIntoPage(inRAT, numIntCols, numFloatCols,
                    startSegId, endSegId)

        # copy any in recodeDict into the new outPagedRat
        copySubsettedSegmentsToNew(inPage, outPagedRat, recodeDict)

        writeCompletedPagesForSubset(inRAT, outRAT, outPagedRat)

    # write out the histogram we've been updating
    histArray = numpy.empty(outRAT.GetRowCount(), dtype=numpy.float64)
    setHistogramFromDictionary(histogramDict, histArray)

    colNum = outRAT.GetColOfUsage(gdal.GFU_PixelCount)
    if colNum == -1:
        outRAT.CreateColumn('Histogram', gdal.GFT_Real, gdal.GFU_PixelCount)
        colNum = outRAT.GetColumnCount() - 1
    outRAT.WriteArray(histArray, colNum)
    del histArray

    # optional column with old segids
    if origSegIdColName is not None:
        # find or create column
        colNum = -1
        for n in range(outRAT.GetColumnCount()):
            if outRAT.GetNameOfCol(n) == origSegIdColName:
                colNum = n
                break

        if colNum == -1:
            outRAT.CreateColumn(origSegIdColName, gdal.GFT_Integer,
                    gdal.GFU_Generic)
            colNum = outRAT.GetColumnCount() - 1

        origSegIdArray = numpy.empty(outRAT.GetRowCount(), dtype=numpy.int32)
        setSubsetRecodeFromDictionary(recodeDict, origSegIdArray)
        outRAT.WriteArray(origSegIdArray, colNum)


@njit
def copySubsettedSegmentsToNew(inPage, outPagedRat, recodeDict):
    """
    Using the recodeDict, copy across the rows inPage to outPage.

    inPage is processed and (taking into account of inPage.startSegId)
    the original input row found. This value is then
    looked up in recodeDict to find the row in the output RAT to
    copy the row from the input to.

    Parameters
    ----------
      inPage : tiling.RatPage
        A page of RAT from the input file
      outPagedRat : numba.typed.Dict
        In-memory pages of the output RAT, as created by createPagedRat().
        This is modified in-place, creating new pages as required.
      recodeDict : numba.typed.Dict
        Keyed by original segment ID, values are the corresponding
        segment IDs in the subset

    """
    numIntCols = inPage.intcols.shape[0]
    numFloatCols = inPage.floatcols.shape[0]
    maxSegId = len(recodeDict)
    for inRowInPage in range(inPage.intcols.shape[1]):
        inRow = tiling.segIdNumbaType(inPage.startSegId + inRowInPage)
        if inRow not in recodeDict:
            # this one is not in this subset, skip
            continue
        outRow = recodeDict[inRow]

        outPageId = tiling.getRatPageId(outRow)
        outRowInPage = outRow - outPageId
        if outPageId not in outPagedRat:
            numSegThisPage = min(tiling.RAT_PAGE_SIZE, (maxSegId - outPageId + 1))
            outPagedRat[outPageId] = tiling.RatPage(numIntCols, numFloatCols,
                            outPageId, numSegThisPage)
            if outPageId == shepseg.SEGNULLVAL:
                # nothing will get written to this one, but needs to be
                # marked as complete so whole page will be written
                outPagedRat[outPageId].setSegmentComplete(shepseg.SEGNULLVAL)

        outPage = outPagedRat[outPageId]
        for n in range(numIntCols):
            outPage.intcols[n, outRowInPage] = inPage.intcols[n, inRowInPage]
        for n in range(numFloatCols):
            outPage.floatcols[n, outRowInPage] = inPage.floatcols[n, inRowInPage]

        # we mark this as complete as we have copied the row over.
        outPage.setSegmentComplete(outRow)


@njit
def setHistogramFromDictionary(dictn, histArray):
    """
    Given a dictionary of pixel counts keyed on index,
    write these values to the array.
    """
    for idx in dictn:
        histArray[idx] = dictn[idx]
    histArray[shepseg.SEGNULLVAL] = 0


@njit
def setSubsetRecodeFromDictionary(dictn, array):
    """
    Given the recodeDict write the original values to the array
    at the new indices.
    """
    for idx in dictn:
        array[dictn[idx]] = idx
    array[shepseg.SEGNULLVAL] = 0


@njit
def readColDataIntoPage(page, data, idx, colType, minVal):
    """
    Numba function to quickly read a column returned by
    rat.ReadAsArray() info a RatPage.
    """
    for i in range(data.shape[0]):
        page.setRatVal(i + minVal, colType, idx, data[i])


def readRATIntoPage(rat, numIntCols, numFloatCols, minVal, maxVal):
    """
    Create a new RatPage() object that represents the section of the RAT
    for a tile of an image. The part of the RAT between minVal and maxVal
    is read in and a RatPage() instance returned with the startSegId param
    set to minVal.

    """
    minVal = int(minVal)
    nrows = int(maxVal - minVal) + 1
    page = tiling.RatPage(numIntCols, numFloatCols, minVal, nrows)

    intColIdx = 0
    floatColIdx = 0
    for col in range(rat.GetColumnCount()):
        dtype = rat.GetTypeOfCol(col)
        data = rat.ReadAsArray(col, start=minVal, length=nrows)
        if dtype == gdal.GFT_Integer:
            readColDataIntoPage(page, data, intColIdx,
                tiling.STAT_DTYPE_INT, minVal)
            intColIdx += 1
        else:
            readColDataIntoPage(page, data, floatColIdx,
                tiling.STAT_DTYPE_FLOAT, minVal)
            floatColIdx += 1

    return page


def copyColumns(inRat, outRat):
    """
    Copy column structure from inRat to outRat. Note that this just creates
    the empty columns in outRat, it does not copy any data.

    Parameters
    ----------
      inRat, outRat : gdal.RasterAttributeTable
        Columns found in inRat are created on outRat

    Returns
    -------
      numIntCols : int
        Number of integer columns found
      numFloatCols : int
        Number of float columns found

    """
    numIntCols = 0
    numFloatCols = 0
    for col in range(inRat.GetColumnCount()):
        dtype = inRat.GetTypeOfCol(col)
        usage = inRat.GetUsageOfCol(col)
        name = inRat.GetNameOfCol(col)
        outRat.CreateColumn(name, dtype, usage)
        if dtype == gdal.GFT_Integer:
            numIntCols += 1
        elif dtype == gdal.GFT_Real:
            numFloatCols += 1
        else:
            raise TypeError("String columns not supported")

    return numIntCols, numFloatCols


@njit
def processSubsetTile(tile, recodeDict, histogramDict, maskData):
    """
    Process a tile of the subset area. Returns a new tile with the new codes.
    Fills in the recodeDict as it goes and also updates histogramDict.

    Parameters
    ----------
      tile : shepseg.SegIdType ndarray (tileNrows, tileNcols)
        Input tile of segment IDs
      recodeDict : numba.typed.Dict
        Keyed by original segment ID, values are the corresponding
        segment IDs in the subset
      histogramDict : numba.typed.Dict
        Histogram counts in the subset, keyed by new segment ID
      maskData : None or int ndarray (tileNrows, tileNcols)
        If not None, then is a raster mask. Only pixels which are
        non-zero in the mask will be included in the subset

    Returns
    -------
      outData : shepseg.SegIdType ndarray (tileNrows, tileNcols)
        Recoded copy of the input tile.

    """
    outData = numpy.zeros_like(tile)

    ysize, xsize = tile.shape
    # go through each pixel
    for y in range(ysize):
        for x in range(xsize):
            segId = tile[y, x]
            if maskData is not None and maskData[y, x] == 0:
                # if this one is masked out - skip
                outData[y, x] = shepseg.SEGNULLVAL
                continue

            if segId == shepseg.SEGNULLVAL:
                # null segment - skip
                outData[y, x] = shepseg.SEGNULLVAL
                continue

            if segId not in recodeDict:
                # haven't encountered this pixel before, generate the new id for it
                outSegId = len(recodeDict) + 1

                # add this new value to our recode dictionary
                recodeDict[segId] = tiling.segIdNumbaType(outSegId)

            # write this new value to the output image
            newval = recodeDict[segId]
            outData[y, x] = newval
            # update histogram
            if newval not in histogramDict:
                histogramDict[newval] = tiling.segIdNumbaType(0)
            histogramDict[newval] = tiling.segIdNumbaType(histogramDict[newval] + 1)

    return outData


def writeCompletedPagesForSubset(inRAT, outRAT, outPagedRat):
    """
    For the subset operation. Writes out any completed pages to outRAT
    using the inRAT as a template.

    Parameters
    ----------
      inRAT, outRAT : gdal.RasterAttributeTable
        The input and output raster attribute tables.
      outPagedRat : numba.typed.Dict
        The paged RAT in memory, as created by createPagedRat()

    """
    for pageId in outPagedRat:
        ratPage = outPagedRat[pageId]
        if ratPage.pageComplete():
            # this one one is complete. Grow RAT if required
            maxRow = ratPage.startSegId + ratPage.intcols.shape[1]
            if outRAT.GetRowCount() < maxRow:
                outRAT.SetRowCount(maxRow)

            # loop through the input RAT, using the type info
            # of each column to decide intcols/floatcols etc
            intColIdx = 0
            floatColIdx = 0
            for col in range(inRAT.GetColumnCount()):
                dtype = inRAT.GetTypeOfCol(col)
                if dtype == gdal.GFT_Integer:
                    data = ratPage.intcols[intColIdx]
                    intColIdx += 1
                else:
                    data = ratPage.floatcols[floatColIdx]
                    floatColIdx += 1

                outRAT.WriteArray(data, col, start=ratPage.startSegId)

            # this one is done
            outPagedRat.pop(pageId)


class PyShepSegSubsetError(Exception):
    pass
