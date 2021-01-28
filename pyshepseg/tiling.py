"""
Routines in support of tiled segmentation of very large rasters. 

This module is still under development. 

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

import os
import time

import numpy
from osgeo import gdal
import scipy.stats

from . import shepseg

def fitSpectralClustersWholeFile(filename, numClusters=60, 
        bandNumbers=None, subsamplePcnt=None, imgNullVal=None, 
        fixedKMeansInit=False):
    """
    Given a raster filename, read a selected sample of pixels
    and use these to fit a spectral cluster model. Uses GDAL
    to read the pixels, and shepseg.fitSpectralClusters() 
    to do the fitting. 
    
    If bandNumbers is not None, this is a list of band numbers 
    (1 is 1st band) to use in fitting the model. 
    
    If subsamplePcnt is not None, this is the percentage of 
    pixels sampled. If it is None, then a suitable subsample is 
    calculated such that around one million pixels are sampled
    (Note - this would include null pixels, so if the image is 
    dominated by nulls, this would undersample.) 
    No further subsampling is carried out by fitSpectralClusters(). 
    
    If imgNullVal is None, the file is queried for a null value. 
    If none is defined there, then no null value is used. If 
    each band has a different null value, an exception is raised. 
    
    fixedKMeansInit is passed to fitSpectralClusters(), see there
    for details. 
    
    Returns a tuple
        (kmeansObj, subsamplePcnt, imgNullVal)
    where kmeansObj is the fitted object, subsamplePcnt
    is the subsample percentage actually used, and imgNullVal 
    is the null value used (perhaps from the file). 
    
    """
    ds = gdal.Open(filename)
    if bandNumbers is None:
        bandNumbers = range(1, ds.RasterCount+1)
    
    if subsamplePcnt is None:
        # We will try to sample roughly this many pixels
        dfltTotalPixels = 1000000
        totalImagePixels = ds.RasterXSize * ds.RasterYSize
        subsampleProp = numpy.sqrt(dfltTotalPixels / totalImagePixels)
        subsamplePcnt = 100 * subsampleProp
    else:
        subsampleProp = subsamplePcnt / 100.0
    
    if imgNullVal is None:
        nullValArr = numpy.array([ds.GetRasterBand(i).GetNoDataValue() 
            for i in bandNumbers])
        if (nullValArr != nullValArr[0]).any():
            raise PyShepSegTilingError("Different null values in some bands")
        imgNullVal = nullValArr[0]
    
    nRows_sub = int(round(ds.RasterYSize * subsampleProp))
    nCols_sub = int(round(ds.RasterXSize * subsampleProp))
    
    bandList = []
    for bandNum in bandNumbers:
        bandObj = ds.GetRasterBand(bandNum)
        band = bandObj.ReadAsArray(buf_xsize=nCols_sub, buf_ysize=nRows_sub)
        bandList.append(band)
    img = numpy.array(bandList)
    
    kmeansObj = shepseg.fitSpectralClusters(img, numClusters=numClusters, 
        subsamplePcnt=100, imgNullVal=imgNullVal, 
        fixedKMeansInit=fixedKMeansInit)
    
    return (kmeansObj, subsamplePcnt, imgNullVal)


def saveKMeansObj(kmeansObj, filename):
    """
    Saves the given KMeans object into the given filename. 
    
    Since the KMeans object is not pickle-able, use our own
    simple JSON form to save the cluster centres. The 
    corresponding function loadKMeansObj() can be used to
    re-create the original object (at least functionally equivalent). 
    """
    # Check that it really is not pickle-able, I am just assuming....
    

def loadKMeansObj(filename):
    """
    Load a KMeans object from a file, as saved by saveKMeansObj(). 
    """

class TileInfo(object):
    """
    Class that holds the pixel coordinates of the tiles within 
    an image. 
    """
    def __init__(self, filename):
        self.filename = filename
        self.tiles = {}
        self.ncols = None
        self.nrows = None
        
    def addTile(self, xpos, ypos, xsize, ysize, col, row):
        self.tiles[(col, row)] = (xpos, ypos, xsize, ysize)
        
    def getNumTiles(self):
        return len(self.tiles)
        
    def getTile(self, col, row):
        return self.tiles[(col, row)]
        
def getTilesForFile(infile, tileSize, overlapSize):
    """
    Return a TileInfo object for a given file and input
    parameters.
    """
    ds = gdal.Open(infile)
    
    # ensure int
    tileSize = int(tileSize)
    overlapSize = int(overlapSize)
    
    tileInfo = TileInfo(infile)
        
    yDone = False
    ypos = 0
    xtile = 0
    ytile = 0
    while not yDone:
    
        xDone = False
        xpos = 0
        xtile = 0
        ysize = tileSize
        if (ypos + ysize) > ds.RasterYSize:
            ysize = ds.RasterYSize - ypos
            yDone = True
            if ysize == 0:
                break
    
        while not xDone:
            xsize = tileSize
            if (xpos + xsize) > ds.RasterXSize:
                xsize = ds.RasterXSize - xpos
                xDone = True
                if xsize == 0:
                    break

            tileInfo.addTile(xpos, ypos, xsize, ysize, xtile, ytile)
            xpos += (tileSize - overlapSize)
            xtile += 1
            
        ypos += (tileSize - overlapSize)
        ytile += 1
        
    tileInfo.ncols = xtile
    tileInfo.nrows = ytile
        
    return tileInfo
    

def doTiledShepherdSegmentation(infile, outfile, tileSize, overlapSize=None,
        minSegmentSize=50, numClusters=60, bandNumbers=None, subsamplePcnt=None, 
        maxSpectralDiff='auto', imgNullVal=None, fixedKMeansInit=False,
        fourConnected=True, verbose=False, simpleTileRecode=False):
    """
    Run the Shepherd segmentation algorithm in a memory-efficient
    manner, suitable for large raster files. Runs the segmentation
    on separate tiles across the raster, then stitches these
    together into a single output segment raster. 
    
    The initial spectral clustering is performed on a sub-sample
    of the whole raster, to create consistent clusters. These are 
    then used as seeds for all individual tiles. 
    
    """
    if (overlapSize % 2) != 0:
        raise PyShepSegTilingError("Overlap size must be an even number")

    kmeansObj, subSamplePcnt, imgNullVal = fitSpectralClustersWholeFile(infile, 
            numClusters, bandNumbers, subsamplePcnt, imgNullVal, fixedKMeansInit)
    
    inDs = gdal.Open(infile)
    
    if overlapSize is None:
        overlapSize = minSegmentSize / 2
        
    tileInfo = getTilesForFile(infile, tileSize, overlapSize)
    
    if bandNumbers is None:
        bandNumbers = range(1, inDs.RasterCount+1)
        
    transform = inDs.GetGeoTransform()
    tileFilenames = {}
    
    for col, row in tileInfo.tiles:
        xpos, ypos, xsize, ysize = tileInfo.getTile(col, row)
        lyrDataList = []
        for bandNum in bandNumbers:
            lyr = inDs.GetRasterBand(bandNum)
            lyrData = lyr.ReadAsArray(xpos, ypos, xsize, ysize)
            lyrDataList.append(lyrData)
            
        img = numpy.array(lyrDataList)
    
        segResult = shepseg.doShepherdSegmentation(img, 
                    minSegmentSize=minSegmentSize,
                    maxSpectralDiff=maxSpectralDiff, imgNullVal=imgNullVal, 
                    fourConnected=fourConnected, kmeansObj=kmeansObj, 
                    verbose=verbose)
        
        filename = 'tile_{}_{}.kea'.format(col, row)
        tileFilenames[(col, row)] = filename
        outDrvr = gdal.GetDriverByName('KEA')
        
        if os.path.exists(filename):
            outDrvr.Delete(filename)

        outType = gdal.GDT_UInt32

        outDs = outDrvr.Create(filename, xsize, ysize, 1, outType)
        outDs.SetProjection(inDs.GetProjection())
        subsetTransform = list(transform)
        subsetTransform[0] = transform[0] + xpos * transform[1]
        subsetTransform[3] = transform[3] + ypos * transform[5]
        outDs.SetGeoTransform(tuple(subsetTransform))
        b = outDs.GetRasterBand(1)
        b.WriteArray(segResult.segimg)
        b.SetMetadataItem('LAYER_TYPE', 'thematic')
        b.SetNoDataValue(shepseg.SEGNULLVAL)
        
        writeRandomColourTable(b, segResult.segimg.max()+1)

        del outDs
        
    stitchTiles(outfile, tileFilenames, tileInfo, overlapSize,
        simpleTileRecode)


def stitchTiles(outfile, tileFilenames, tileInfo, overlapSize,
        simpleTileRecode):
    """
    Recombine individual tiles into a single segment raster output 
    file. Segment ID values are recoded to be unique across the whole
    raster, and contiguous. 
    
    outfile is the name of the final output raster. 
    tileFilenames is a dictionary of the individual tile filenames, 
    keyed by a tuple of (col, row) defining which tile it is. 
    tileInfo is the object returned by getTilesForFile. 
    overlapSize is the number of pixels in the overlap between tiles. 
    
    If simpleTileRecode is True, a simpler method will be used to 
    recode segment IDs, using just a block offset to shift ID numbers.
    If it is False, then a more complicated method is used which
    recodes and merges segments which cross the boundary between tiles. 
    
    """
    marginSize = int(overlapSize / 2)

    inDs = gdal.Open(tileInfo.filename)

    outDrvr = gdal.GetDriverByName('KEA')

    if os.path.exists(outfile):
        outDrvr.Delete(outfile)

    outType = gdal.GDT_UInt32

    outDs = outDrvr.Create(outfile, inDs.RasterXSize, inDs.RasterYSize, 1, outType)
    outDs.SetProjection(inDs.GetProjection())
    outDs.SetGeoTransform(inDs.GetGeoTransform())
    outBand = outDs.GetRasterBand(1)
    outBand.SetMetadataItem('LAYER_TYPE', 'thematic')
    outBand.SetNoDataValue(shepseg.SEGNULLVAL)
    
    colRows = sorted(tileInfo.tiles.keys())                
    maxSegId = 0
    
    for col, row in colRows:
        
        filename = tileFilenames[(col, row)]
        ds = gdal.Open(filename)
        print('reading', col, row)
        tileData = ds.ReadAsArray()
        xpos, ypos, xsize, ysize = tileInfo.getTile(col, row)
        
        top = marginSize
        bottom = ysize - marginSize
        left = marginSize
        right = xsize - marginSize
        
        xout = xpos + marginSize
        yout = ypos + marginSize

        rightName = overlapFilename(col, row, RIGHT_OVERLAP)
        bottomName = overlapFilename(col, row, BOTTOM_OVERLAP)
        
        if row == 0:
            top = 0
            yout = ypos

        if row == (tileInfo.nrows-1):
            bottom = ysize
            bottomName = None
            
        if col == 0:
            left = 0
            xout = xpos
            
        if col == (tileInfo.ncols-1):
            right = xsize
            rightName = None
        
        if simpleTileRecode:
            nullmask = (tileData == shepseg.SEGNULLVAL)
            tileData += maxSegId
            tileData[nullmask] = shepseg.SEGNULLVAL
        else:
            t0 = time.time()
            tileData = recodeTile(tileData, maxSegId, row, col, overlapSize)
            print('recode {:.2f} seconds'.format(time.time()-t0))
        
        print('writing', col, row)
        tileDataTrimmed = tileData[top:bottom, left:right]
        outBand.WriteArray(tileDataTrimmed, xout, yout)

        if rightName is not None:
            numpy.save(rightName, tileData[:, -overlapSize:])
        if bottomName is not None:
            numpy.save(bottomName, tileData[-overlapSize:, :])    
        
        nonNull = (tileDataTrimmed != shepseg.SEGNULLVAL)
        maxSegId = tileDataTrimmed[nonNull].max()

    print('write colour table', maxSegId, type(maxSegId))    
    nRows = maxSegId+1

    writeRandomColourTable(outBand, nRows)


RIGHT_OVERLAP = 'right'
BOTTOM_OVERLAP = 'bottom'
def overlapFilename(col, row, edge):
    """
    Return the filename used for the overlap array
    """
    return '{}_{}_{}.npy'.format(edge, col, row)


# The two orientations of the overlap region
HORIZONTAL = 0
VERTICAL = 1

def recodeTile(tileData, maxSegId, tileRow, tileCol, overlapSize):
    """
    Adjust the segment ID numbers in the current tile, 
    to make them globally unique across the whole mosaic.
    
    Make use of the overlapping regions of tiles above and left,
    to identify shared segments, and recode those to segment IDs 
    from the adjacent tiles (i.e. we change the current tile, not 
    the adjacent ones). Non-shared segments are increased so they 
    are larger than previous values. 
    
    tileData is the array of segment IDs for a single tile. 
    maxSegId is the current maximum segment ID for all preceding
    tiles. 
    tileRow, tileCol are the row/col numbers of this tile, within
    the whole-mosaic tile numbering scheme. 
    
    Return a copy of tileData, with new segment ID numbers. 
    
    """
    # The A overlaps are from the current tile. The B overlaps 
    # are the same regions from the adjacent tiles, and we load 
    # them here from the saved .npy files. 
    topOverlapA = tileData[:overlapSize, :]
    leftOverlapA = tileData[:, :overlapSize]
    
    recodeDict = {}    

    # Read in the bottom and right regions of the adjacent tiles
    if tileRow > 0:
        topOverlapFilename = overlapFilename(tileCol, tileRow-1, BOTTOM_OVERLAP)
        topOverlapB = numpy.load(topOverlapFilename)

        recodeSharedSegments(tileData, topOverlapA, topOverlapB, 
            HORIZONTAL, recodeDict)

    if tileCol > 0:
        leftOverlapFilename = overlapFilename(tileCol-1, tileRow, RIGHT_OVERLAP)
        leftOverlapB = numpy.load(leftOverlapFilename)

        recodeSharedSegments(tileData, leftOverlapA, leftOverlapB, 
            VERTICAL, recodeDict)
    
    (newTileData, newMaxSegId) = relabelSegments(tileData, recodeDict, 
        maxSegId)
    
    return newTileData


def recodeSharedSegments(tileData, overlapA, overlapB, orientation,
        recodeDict):
    """
    """
    # The current segment IDs just from the overlap region.
    segIdList = numpy.unique(overlapA)
    # Ensure that we do not include the null segment ID
    segIdList = segIdList[segIdList!=shepseg.SEGNULLVAL]
    
    segSize = shepseg.makeSegSize(overlapA)
    segLoc = shepseg.makeSegmentLocations(overlapA, segSize)
    
    # Find the segments which cross the stitch line
    segsOnStitch = []
    for segid in segIdList:
        if crossesMidline(overlapA, segLoc[segid], orientation):
            segsOnStitch.append(segid)
    
    for segid in segsOnStitch:
        # Get the pixel row and column numbers of every pixel
        # in this segment. Note that because we are using
        # the top and left overlap regions, the pixel row/col 
        # numbers in the full tile are identical to those for 
        # the corresponding pixels in the overlap region arrays
        segNdx = segLoc[segid].getSegmentIndices()
        
        # Find the most common segment ID in the corresponding
        # pixels from the B overlap array. In principle there 
        # should only be one, but just in case. 
        modeObj = scipy.stats.mode(overlapB[segNdx])
        segIdFromB = modeObj.mode[0]
        
        # Now record this recoding relationship
        recodeDict[segid] = segIdFromB


def relabelSegments(tileData, recodeDict, maxSegId):
    """
    Recode the segment IDs in the given tileData array.
    
    For segment IDs which are keys in recodeDict, these
    are replaced with the corresponding entry. For all other 
    segment IDs, they are replaced with sequentially increasing
    ID numbers, starting from one more than the previous
    maximum segment ID (maxSegId). 
    
    A re-coded copy of tileData is createded, the original is 
    unchanged. 
    
    Return value is a tuple
        (newTileData, newMaxSegId)
    
    """
    newTileData = numpy.full(tileData.shape, shepseg.SEGNULLVAL, 
        dtype=tileData.dtype)
    
    segSize = shepseg.makeSegSize(tileData)
    segLoc = shepseg.makeSegmentLocations(tileData, segSize)

    newSegId = maxSegId
    oldSegmentIDs = segLoc.keys()
    
    for segid in oldSegmentIDs:
        pixNdx = segLoc[shepseg.SegIdType(segid)].getSegmentIndices()
        if segid in recodeDict:
            newTileData[pixNdx] = recodeDict[segid]
        else:
            newSegId += 1
            newTileData[pixNdx] = newSegId
    
    return (newTileData, newSegId)


def crossesMidline(overlap, segLoc, orientation):
    """
    Return True if the given segment crosses the midline of the
    overlap array. Orientation of the midline is either
        HORIZONTAL or VERTICAL
        
    segLoc is the segment location entry for the segment in question
    
    """
    (nrows, ncols) = overlap.shape
    if orientation == HORIZONTAL:
        mid = int(nrows / 2)
        n = 0
    elif orientation == VERTICAL:
        mid = int(ncols / 2)
        n = 1

    minN = segLoc.rowcols[:, n].min()
    maxN = segLoc.rowcols[:, n].max()
    
    return ((minN <= mid) & (maxN > mid))


def writeRandomColourTable(outBand, nRows):

    nRows = int(nRows)
    colNames = ["Blue", "Green", "Red"]
    colUsages = [gdal.GFU_Blue, gdal.GFU_Green, gdal.GFU_Red]

    attrTbl = outBand.GetDefaultRAT()
    attrTbl.SetRowCount(nRows)
    
    for band in range(3):
        attrTbl.CreateColumn(colNames[band], gdal.GFT_Integer, colUsages[band])
        colNum = attrTbl.GetColumnCount() - 1
        colour = numpy.random.random_integers(0, 255, size=nRows)
        attrTbl.WriteArray(colour, colNum)
        
    alpha = numpy.full((nRows,), 255, dtype=numpy.uint8)
    alpha[shepseg.SEGNULLVAL] = 0
    attrTbl.CreateColumn('Alpha', gdal.GFT_Integer, gdal.GFU_Alpha)
    colNum = attrTbl.GetColumnCount() - 1
    attrTbl.WriteArray(alpha, colNum)

class PyShepSegTilingError(Exception): pass
