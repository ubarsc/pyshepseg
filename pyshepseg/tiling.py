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
import shutil
import tempfile

import numpy
from osgeo import gdal
gdal.UseExceptions()
import scipy.stats

from numba import njit

from . import shepseg

TEMPFILES_DRIVER = 'KEA'
TEMPFILES_EXT = 'kea'

DFLT_TILESIZE = 4096
DFLT_OVERLAPSIZE = 200


class TiledSegmentationResult(object):
    """
    Result of tiled segmentation
    
    Attributes:
      maxSegId: Largest segment ID used in final segment image
      numTileRows: Number of rows of tiles used
      numTileCols: Number of columns of tiles used
      subSamplePcnt: Percentage of image subsampled for clustering
      maxSpectralDiff: The value used to limit segment merging (in all tiles)
      kmeans: The sklearn KMeans object, after fitting
      
    """
    maxSegId = None
    numTileRows = None
    numTileCols = None
    subSamplePcnt = None
    maxSpectralDiff = None
    kmeans = None


def fitSpectralClustersWholeFile(inDs, bandNumbers, numClusters=60, 
        subsamplePcnt=None, imgNullVal=None, 
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
    if subsamplePcnt is None:
        # We will try to sample roughly this many pixels
        dfltTotalPixels = 1000000
        totalImagePixels = inDs.RasterXSize * inDs.RasterYSize
        # subsampleProp is the proportion of rows and columns sampled
        subsampleProp = numpy.sqrt(dfltTotalPixels / totalImagePixels)
        # subsamplePcnt is the percentage of total pixels sampled
        subsamplePcnt = 100 * subsampleProp**2
    else:
        # subsampleProp is the proportion of rows and columns sampled,
        # hence we sqrt
        subsampleProp = numpy.sqrt(subsamplePcnt / 100.0)
    
    if imgNullVal is None:
        nullValArr = numpy.array([inDs.GetRasterBand(i).GetNoDataValue() 
            for i in bandNumbers])
        if (nullValArr != nullValArr[0]).any():
            raise PyShepSegTilingError("Different null values in some bands")
        imgNullVal = nullValArr[0]
    
    bandList = []
    for bandNum in bandNumbers:
        bandObj = inDs.GetRasterBand(bandNum)
        band = readSubsampledImageBand(bandObj, subsampleProp)
        bandList.append(band)
    img = numpy.array(bandList)
    
    kmeansObj = shepseg.fitSpectralClusters(img, numClusters=numClusters, 
        subsamplePcnt=100, imgNullVal=imgNullVal, 
        fixedKMeansInit=fixedKMeansInit)
    
    return (kmeansObj, subsamplePcnt, imgNullVal)


def readSubsampledImageBand(bandObj, subsampleProp):
    """
    Read in a sub-sampled copy of the whole of the given band. 
    
    bandObj is an open gdal.Band object. 
    subsampleProp is the proportion by which to sub-sample 
    (i.e. a value between zero and 1, applied to rows and
    columns separately)
    
    Returns a numpy array of the image data, equivalent to 
    gdal.Band.ReadAsArray(). 
    
    Note that one can, in principle, do this directly using GDAL. 
    However, if overview layers are present in the file, it will use
    these, and so is dependent on how these were created. Since 
    these are often created just for display purposes, past experience
    has shown that they are not always to be trusted as data, 
    so we have chosen to always go directly to the full resolution 
    image. 
    
    """
    # A skip factor, applied to rows and column
    skip = int(round(1./subsampleProp))
    
    tileSize = 1024
    (nlines, npix) = (bandObj.YSize, bandObj.XSize)
    numXtiles = int(numpy.ceil(npix / tileSize))
    numYtiles = int(numpy.ceil(nlines / tileSize))

    tileRowList = []

    for tileRow in range(numYtiles):
        ypos = tileRow * tileSize
        ysize = min(tileSize, (nlines - ypos))
        
        tileColList = []
        for tileCol in range(numXtiles):
            xpos = tileCol * tileSize
            xsize = min(tileSize, (npix - xpos))
            
            tile = bandObj.ReadAsArray(xpos, ypos, xsize, ysize)
            
            tileSub = tile[::skip, ::skip]
            tileColList.append(tileSub)
        
        tileRow = numpy.concatenate(tileColList, axis=1)
        tileRowList.append(tileRow)
    
    imgSub = numpy.concatenate(tileRowList, axis=0)
    return imgSub


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
    def __init__(self):
        self.tiles = {}
        self.ncols = None
        self.nrows = None
        
    def addTile(self, xpos, ypos, xsize, ysize, col, row):
        self.tiles[(col, row)] = (xpos, ypos, xsize, ysize)
        
    def getNumTiles(self):
        return len(self.tiles)
        
    def getTile(self, col, row):
        return self.tiles[(col, row)]
        
def getTilesForFile(ds, tileSize, overlapSize):
    """
    Return a TileInfo object for a given file and input
    parameters.
    """
    # ensure int
    tileSize = int(tileSize)
    overlapSize = int(overlapSize)
    
    tileInfo = TileInfo()
        
    yDone = False
    ypos = 0
    xtile = 0
    ytile = 0
    while not yDone:
    
        xDone = False
        xpos = 0
        xtile = 0
        ysize = tileSize
        # ensure that we can fit another whole tile before the edge
        # - grow this tile if needed so we don't end up with slivers
        if (ypos + ysize*2) > ds.RasterYSize:
            ysize = ds.RasterYSize - ypos
            yDone = True
            if ysize == 0:
                break
    
        while not xDone:
            xsize = tileSize
            # ensure that we can fit another whole tile before the edge
            # - grow this tile if needed so we don't end up with slivers
            if (xpos + xsize*2) > ds.RasterXSize:
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
    

def doTiledShepherdSegmentation(infile, outfile, tileSize=DFLT_TILESIZE, 
        overlapSize=DFLT_OVERLAPSIZE, minSegmentSize=50, numClusters=60, 
        bandNumbers=None, subsamplePcnt=None, maxSpectralDiff='auto', 
        imgNullVal=None, fixedKMeansInit=False, fourConnected=True, 
        verbose=False, simpleTileRecode=False, outputDriver='KEA',
        spectDistPcntile=50):
    """
    Run the Shepherd segmentation algorithm in a memory-efficient
    manner, suitable for large raster files. Runs the segmentation
    on separate tiles across the raster, then stitches these
    together into a single output segment raster. 
    
    The initial spectral clustering is performed on a sub-sample
    of the whole raster (using fitSpectralClustersWholeFile), 
    to create consistent clusters. These are then used as seeds 
    for all individual tiles. 
    
    The tileSize is the minimum width/height of the tiles (in pixels).
    These tiles are overlapped by overlapSize (also in pixels), both 
    horizontally and vertically.
    Tiles on the right and bottom edges of the input image may end up 
    slightly larger than tileSize to ensure there are no small tiles.
    
    Most of the arguments are passed through to 
    shepseg.doShepherdSegmentation, and are described in the docstring 
    for that function. 
    
    Return an instance of TiledSegmentationResult class. 
    
    """
    if verbose:
        print("Starting tiled segmentation")
    if (overlapSize % 2) != 0:
        raise PyShepSegTilingError("Overlap size must be an even number")

    inDs = gdal.Open(infile)

    if bandNumbers is None:
        bandNumbers = range(1, inDs.RasterCount+1)

    t0 = time.time()
    kmeansObj, subSamplePcnt, imgNullVal = fitSpectralClustersWholeFile(inDs, 
            bandNumbers, numClusters, subsamplePcnt, imgNullVal, fixedKMeansInit)
    if verbose:
        print("KMeans of whole raster {:.2f} seconds".format(time.time()-t0))
        print("Subsample Percentage={:.2f}".format(subSamplePcnt))
    
    # create a temp directory for use in splitting out tiles, overlaps etc
    tempDir = tempfile.mkdtemp()
    
    tileInfo = getTilesForFile(inDs, tileSize, overlapSize)
    if verbose:
        print("Found {} tiles, with {} rows and {} cols".format(
            tileInfo.getNumTiles(), tileInfo.nrows, tileInfo.ncols))
        
    transform = inDs.GetGeoTransform()
    tileFilenames = {}

    outDrvr = gdal.GetDriverByName(TEMPFILES_DRIVER)

    if outDrvr is None:
        msg = 'This GDAL does not support driver {}'.format(TEMPFILES_DRIVER)
        raise SystemExit(msg)
    
    colRowList = sorted(tileInfo.tiles.keys(), key=lambda x:(x[1], x[0]))
    tileNum = 1
    for col, row in colRowList:
        if verbose:
            print("\nDoing tile {} of {}: row={}, col={}".format(
                tileNum, len(colRowList), row, col))

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
                    verbose=verbose, spectDistPcntile=spectDistPcntile)
        
        filename = 'tile_{}_{}.{}'.format(col, row, TEMPFILES_EXT)
        filename = os.path.join(tempDir, filename)
        tileFilenames[(col, row)] = filename
        
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
        
        del outDs
        tileNum += 1
        
    maxSegId = stitchTiles(inDs, outfile, tileFilenames, tileInfo, overlapSize,
        tempDir, simpleTileRecode, outputDriver, verbose)
        
    shutil.rmtree(tempDir)

    tiledSegResult = TiledSegmentationResult()
    tiledSegResult.maxSegId = maxSegId
    tiledSegResult.numTileRows = tileInfo.nrows
    tiledSegResult.numTileCols = tileInfo.ncols
    tiledSegResult.subSamplePcnt = subSamplePcnt
    tiledSegResult.maxSpectralDiff = segResult.maxSpectralDiff
    tiledSegResult.kmeans = kmeansObj
    
    return tiledSegResult


def stitchTiles(inDs, outfile, tileFilenames, tileInfo, overlapSize,
        tempDir, simpleTileRecode, outputDriver, verbose):
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
    
    Return the maximum segment ID used. 
    
    """
    marginSize = int(overlapSize / 2)

    outDrvr = gdal.GetDriverByName(outputDriver)
    if outDrvr is None:
        msg = 'This GDAL does not support driver {}'.format(outputDriver)
        raise SystemExit(msg)

    if os.path.exists(outfile):
        outDrvr.Delete(outfile)

    outType = gdal.GDT_UInt32

    outDs = outDrvr.Create(outfile, inDs.RasterXSize, inDs.RasterYSize, 1, outType)
    outDs.SetProjection(inDs.GetProjection())
    outDs.SetGeoTransform(inDs.GetGeoTransform())
    outBand = outDs.GetRasterBand(1)
    outBand.SetMetadataItem('LAYER_TYPE', 'thematic')
    outBand.SetNoDataValue(shepseg.SEGNULLVAL)
    
    colRows = sorted(tileInfo.tiles.keys(), key=lambda x:(x[1], x[0]))
    maxSegId = 0
    
    if verbose:
        print("Stitching tiles together")
    reportedRow = -1
    for col, row in colRows:
        if verbose and row != reportedRow:
            print("Stitching tile row {}".format(row))
        reportedRow = row

        filename = tileFilenames[(col, row)]
        ds = gdal.Open(filename)
        tileData = ds.ReadAsArray()
        xpos, ypos, xsize, ysize = tileInfo.getTile(col, row)
        
        top = marginSize
        bottom = ysize - marginSize
        left = marginSize
        right = xsize - marginSize
        
        xout = xpos + marginSize
        yout = ypos + marginSize

        rightName = overlapFilename(col, row, RIGHT_OVERLAP, tempDir)
        bottomName = overlapFilename(col, row, BOTTOM_OVERLAP, tempDir)
        
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
            tileData = recodeTile(tileData, maxSegId, row, col, 
                        overlapSize, tempDir, top, bottom, left, right)
            
        tileDataTrimmed = tileData[top:bottom, left:right]
        outBand.WriteArray(tileDataTrimmed, xout, yout)

        if rightName is not None:
            numpy.save(rightName, tileData[:, -overlapSize:])
        if bottomName is not None:
            numpy.save(bottomName, tileData[-overlapSize:, :])    
        
        tileMaxSegId = tileDataTrimmed.max()
        maxSegId = max(maxSegId, tileMaxSegId)

    return maxSegId


RIGHT_OVERLAP = 'right'
BOTTOM_OVERLAP = 'bottom'
def overlapFilename(col, row, edge, tempDir):
    """
    Return the filename used for the overlap array
    """
    fname = '{}_{}_{}.npy'.format(edge, col, row)
    return os.path.join(tempDir, fname)


# The two orientations of the overlap region
HORIZONTAL = 0
VERTICAL = 1

def recodeTile(tileData, maxSegId, tileRow, tileCol, overlapSize, tempDir,
        top, bottom, left, right):
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
        topOverlapFilename = overlapFilename(tileCol, tileRow-1, 
                                BOTTOM_OVERLAP, tempDir)
        topOverlapB = numpy.load(topOverlapFilename)

        recodeSharedSegments(tileData, topOverlapA, topOverlapB, 
            HORIZONTAL, recodeDict)

    if tileCol > 0:
        leftOverlapFilename = overlapFilename(tileCol-1, tileRow, 
                                RIGHT_OVERLAP, tempDir)
        leftOverlapB = numpy.load(leftOverlapFilename)

        recodeSharedSegments(tileData, leftOverlapA, leftOverlapB, 
            VERTICAL, recodeDict)
    
    (newTileData, newMaxSegId) = relabelSegments(tileData, recodeDict, 
        maxSegId, top, bottom, left, right)
    
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


def relabelSegments(tileData, recodeDict, maxSegId, 
        top, bottom, left, right):
    """
    Recode the segment IDs in the given tileData array.
    
    For segment IDs which are keys in recodeDict, these
    are replaced with the corresponding entry. For all other 
    segment IDs, they are replaced with sequentially increasing
    ID numbers, starting from one more than the previous
    maximum segment ID (maxSegId). 
    
    A re-coded copy of tileData is created, the original is 
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
            (segRows, segCols) = pixNdx
            segLeft = segCols.min()
            segTop = segRows.min()
            # Avoid incrementing newSegId for segments which are
            # handled in neighbouring tiles. For the left and top 
            # margins, the segment must be entirely within the 
            # trimmed tile, while for the right and bottom
            # margins the segment must be at least partially
            # within the trimmed tile. 
            if ((segLeft >= left) and (segTop >= top) and
                    (segLeft < right) and (segTop < bottom)):
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
    
    return ((minN < mid) & (maxN >= mid))


def calcHistogramTiled(segfile, maxSegId, writeToRat=True):
    """
    Calculate a histogram of the given segment image file. 
    
    Note that we need this function because GDAL's GetHistogram
    function does not seem to work when attempting a histogram
    with very large numbers of entries. We want an entry for
    every segment, rather than an approximate count for a range of 
    segment values, and the number of segments is very large. So,
    we need to write our own routine. 
    
    It works in tiles across the image, so that it can process 
    very large images in a memory-efficient way. For the same 
    reason, it keeps the temporary histogram on disk while 
    accumulating it. 
    
    For a raster which can easily fit into memory, a histogram
    can be calculated directly using 
        pyshepseg.shepseg.makeSegSize()
    
    Once completed, the histogram can be written to the image file's
    raster attribute table, if writeToRat is True). It will also be
    returned as a numpy array, indexed by segment ID. 

    segfile can be either a filename string, or an open 
    gdal.Dataset object. If writeToRat is True, then a Dataset
    object should be opened for update. 
    
    """
    # This is the histogram array, indexed by segment ID. 
    # Currently just in memory, it could be quite large, 
    # depending on how many segments there are.
    hist = numpy.zeros((maxSegId+1), dtype=numpy.uint32)
    
    # Open the file
    if isinstance(segfile, gdal.Dataset):
        ds = segfile
    else:
        ds = gdal.Open(segfile, gdal.GA_Update)
    segband = ds.GetRasterBand(1)
    
    tileSize = 1024
    (nlines, npix) = (segband.YSize, segband.XSize)
    numXtiles = int(numpy.ceil(npix / tileSize))
    numYtiles = int(numpy.ceil(nlines / tileSize))

    for tileRow in range(numYtiles):
        for tileCol in range(numXtiles):
            topLine = tileRow * tileSize
            leftPix = tileCol * tileSize
            xsize = min(tileSize, npix-leftPix)
            ysize = min(tileSize, nlines-topLine)
            
            tileData = segband.ReadAsArray(leftPix, topLine, xsize, ysize)
            updateCounts(tileData, hist)

    # Set the histogram count for the null segment to zero
    hist[shepseg.SEGNULLVAL] = 0

    if writeToRat:
        attrTbl = segband.GetDefaultRAT()
        numTableRows = int(maxSegId + 1)
        if attrTbl.GetRowCount() != numTableRows:
            attrTbl.SetRowCount(numTableRows)
        attrTbl.CreateColumn('Histogram', gdal.GFT_Integer, gdal.GFU_PixelCount)
        colNum = attrTbl.GetColumnCount() - 1
        attrTbl.WriteArray(hist, colNum)

    return hist


@njit
def updateCounts(tileData, hist):
    """
    Fast function to increment counts for each segment ID
    """
    (nrows, ncols) = tileData.shape
    for i in range(nrows):
        for j in range(ncols):
            segid = tileData[i, j]
            hist[segid] += 1


class PyShepSegTilingError(Exception): pass
