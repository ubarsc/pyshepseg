"""
Routines in support of tiled segmentation of very large rasters. 

Main entry routine is doTiledShepherdSegmentation(). See that
function for further details. 

The broad idea is that the Shepherd segmentation algorithm, as
implemented in the shepseg module, runs entirely in memory. 
For larger raster files, it is more efficient to divide the raster 
into tiles, segment each tile individually, and stitch the
results together to create a segmentation of the whole raster. 

The main caveats arise from the fact that the initial clustering
is performed on a uniform subsample of the whole image, in 
order to give consistent segment boundaries at tile intersections. 
This means that for a larger raster, with a greater range of 
spectra, one may wish to increase the number of clusters in order 
to allow sufficient initial segments to characterize the variation. 

Related to this, one may also consider reducing the percentile
used for automatic estimation of maxSpectralDiff (see 
shepseg.doShepherdSegmentation() and shepseg.autoMaxSpectralDiff() 
for further details). 

Because of these caveats, one should be very cautious about 
segmenting something like a continental-scale image. There is a 
lot of spectral variation across an area like a whole continent, 
and it may be unwise to use all the same parameters for the
whole area. 

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
from osgeo import osr
import scipy.stats

from numba import njit
from numba.core import types
from numba.typed import Dict
from numba.experimental import jitclass

from . import shepseg

TEMPFILES_DRIVER = 'KEA'
TEMPFILES_EXT = 'kea'

DFLT_TILESIZE = 4096
DFLT_OVERLAPSIZE = 200

DFLT_CHUNKSIZE = 100000


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
        creationOptions=[], spectDistPcntile=50):
    """
    Run the Shepherd segmentation algorithm in a memory-efficient
    manner, suitable for large raster files. Runs the segmentation
    on separate tiles across the raster, then stitches these
    together into a single output segment raster. 
    
    The initial spectral clustering is performed on a sub-sample
    of the whole raster (using fitSpectralClustersWholeFile), 
    to create consistent clusters. These are then used as seeds 
    for all individual tiles. Note that subsamplePcnt is used at 
    this stage, over the whole raster, and is not passed through to 
    shepseg.doShepherdSegmentation() for any further sub-sampling. 
    
    The tileSize is the minimum width/height of the tiles (in pixels).
    These tiles are overlapped by overlapSize (also in pixels), both 
    horizontally and vertically.
    Tiles on the right and bottom edges of the input image may end up 
    slightly larger than tileSize to ensure there are no small tiles.

    outputDriver is a string of the name of the GDAL driver to use
    for the output file. 
    
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

        outDs = outDrvr.Create(filename, xsize, ysize, 1, outType, 
                    options=creationOptions)
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
        tempDir, simpleTileRecode, outputDriver, creationOptions, verbose)
        
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
        tempDir, simpleTileRecode, outputDriver, creationOptions, verbose):
    """
    Recombine individual tiles into a single segment raster output 
    file. Segment ID values are recoded to be unique across the whole
    raster, and contiguous. 
    
    outfile is the name of the final output raster. 
    tileFilenames is a dictionary of the individual tile filenames, 
    keyed by a tuple of (col, row) defining which tile it is. 
    tileInfo is the object returned by getTilesForFile. 
    overlapSize is the number of pixels in the overlap between tiles. 
    outputDriver is a string of the name of the GDAL driver to use
    for the output file. 
    
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

    outDs = outDrvr.Create(outfile, inDs.RasterXSize, inDs.RasterYSize, 1, 
                outType, creationOptions)
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
    Work out a mapping which recodes segment ID numbers from
    the tile in tileData. Segments to be recoded are those which 
    are in the overlap with an earlier tile, and which cross the 
    midline of the overlap, which is where the stitchline between 
    the tiles will fall. 
    
    Updates recodeDict, which is a dictionary keyed on the 
    existing segment ID numbers, where the value of each entry 
    is the segment ID number from the earlier tile, to be used 
    to recode the segment in the current tile. 
    
    overlapA and overlapB are numpy arrays of the overlap region
    in question, giving the segment ID numbers is the two tiles. 
    The values in overlapA are from the earlier tile, and those in 
    overlapB are from the current tile. 
    
    It is critically important that the overlapping region is either
    at the top or the left of the current tile, as this means that 
    the row and column numbers of pixels in the overlap arrays 
    match the same pixels in the full tile. This cannot be used
    for overlaps on the right or bottom of the current tile. 
    
    The orientation parameter defines whether we are dealing with 
    overlap at the top (orientation == HORIZONTAL) or the left
    (orientation == VERTICAL). 
    
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
    
def calcPerSegmentStatsTiled(imgfile, imgbandnum, segfile, 
            statsSelection):
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
        'min', 'max', 'mean', 'stddev', 'median', 'mode', 'percentile'.
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

    """
    segds = segfile
    if not isinstance(segds, gdal.Dataset):
        segds = gdal.Open(segfile, gdal.GA_Update)
    segband = segds.GetRasterBand(1)

    imgds = imgfile
    if not isinstance(imgds, gdal.Dataset):
        imgds = gdal.Open(imgfile, gdal.GA_Update)
    imgband = imgds.GetRasterBand(imgbandnum)
    if (imgband.DataType == gdal.GDT_Float32 or 
            imgband.DataType == gdal.GDT_Float64):
        raise PyShepSegTilingError("Float image types not supported")
        
    if segband.XSize != imgband.XSize or segband.YSize != imgband.YSize:
        raise PyShepSegTilingError("Images must be same size")
        
    if segds.GetGeoTransform() != imgds.GetGeoTransform():
        raise PyShepSegTilingError("Images must have same spatial extent and pixel size")
        
    if not equalProjection(segds.GetProjection(), imgds.GetProjection()):
        raise PyShepSegTilingError("Images must be in the same projection")
    
    attrTbl = segband.GetDefaultRAT()
    existingColNames = [attrTbl.GetNameOfCol(i) 
        for i in range(attrTbl.GetColumnCount())]
        
    histColNdx = checkHistColumn(existingColNames)
    segSize = attrTbl.ReadAsArray(histColNdx).astype(numpy.uint32)
    
    # Create columns, as required
    colIndexList = createStatColumns(statsSelection, attrTbl, existingColNames)
    (statsSelection_fast, numIntCols, numFloatCols) = (
        makeFastStatsSelection(colIndexList, statsSelection))

    # Loop over all tiles in image
    tileSize = 1024
    (nlines, npix) = (segband.YSize, segband.XSize)
    numXtiles = int(numpy.ceil(npix / tileSize))
    numYtiles = int(numpy.ceil(nlines / tileSize))
    
    segDict = createSegDict()
    pagedRat = createPagedRat()
    
    for tileRow in range(numYtiles):
        for tileCol in range(numXtiles):
            topLine = tileRow * tileSize
            leftPix = tileCol * tileSize
            xsize = min(tileSize, npix-leftPix)
            ysize = min(tileSize, nlines-topLine)
            
            tileSegments = segband.ReadAsArray(leftPix, topLine, xsize, ysize)
            tileImageData = imgband.ReadAsArray(leftPix, topLine, xsize, ysize)
            
            accumulateSegDict(segDict, tileSegments, tileImageData)
            calcStatsForCompletedSegs(segDict, pagedRat, statsSelection_fast, 
                segSize, numIntCols, numFloatCols)
            
            writeCompletePages(pagedRat, attrTbl, statsSelection_fast)


# This type is used for all numba jit-ed data which is supposed to 
# match the data type of the imagery pixels. Int64 should be enough
# to hold any integer type, signed or unsigned, up to uint32. 
numbaTypeForImageType = types.int64
# This is the numba equivalent type of shepseg.SegIdType
segIdNumbaType = types.uint32

@njit
def accumulateSegDict(segDict, tileSegments, tileImageData):
    """
    Accumulate per-segment histogram counts for all 
    pixels in the given tile. Updates segDict entries in-place. 
    """
    ysize, xsize = tileSegments.shape
    
    for y in range(ysize):
        for x in range(xsize):
            segId = tileSegments[y, x]
            if segId != shepseg.SEGNULLVAL:
                if segId not in segDict:
                    segDict[segId] = Dict.empty(key_type=numbaTypeForImageType, 
                        value_type=types.uint32)
                
                imgVal = tileImageData[y, x]

                d = segDict[segId]
                imgVal_typed = numbaTypeForImageType(imgVal)
                if imgVal_typed not in d:
                    d[imgVal_typed] = types.uint32(0)

                d[imgVal_typed] = types.uint32(d[imgVal_typed] + 1)


@njit
def checkSegComplete(segDict, segSize, segId):
    """
    Return True if the given segment has a complete entry
    in the segDict, meaning that the pixel count is equal to
    the segment size
    """
    d = segDict[segIdNumbaType(segId)]
    count = 0
    for pixVal in d:
        count += d[pixVal]
    return (count == segSize[segId])


@njit
def calcStatsForCompletedSegs(segDict, pagedRat, statsSelection_fast, segSize,
        numIntCols, numFloatCols):
    """
    Calculate statistics for all complete segments in the segDict.
    Update the pagedRat with the resulting entries. Completed segments
    are then removed from segDict. 
    """
    numStats = len(statsSelection_fast)
    maxSegId = len(segSize) - 1
    segDictKeys = numpy.empty(len(segDict), dtype=segIdNumbaType)
    i = 0
    for segId in segDict:
        segDictKeys[i] = segId
        i += 1
    for segId in segDictKeys:
        segComplete = checkSegComplete(segDict, segSize, segId)
        if segComplete:
            segStats = SegmentStats(segDict[segId])
            ratPageId = getRatPageId(segId)
            if ratPageId not in pagedRat:
                numSegThisPage = min(RAT_PAGE_SIZE, (maxSegId - ratPageId + 1))
                pagedRat[ratPageId] = RatPage(numIntCols, numFloatCols, 
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
            segDict.pop(segIdNumbaType(segId))


def createSegDict():
    """
    Create the Dict of Dicts for handling per-segment histograms. 
    Each entry is a dictionary, and the key is a segment ID.
    Each dictionary within this is the per-segment histogram for
    a single segment. Each of its entries is for a single value from 
    the imagery, the key is the pixel value, and the dictionary value 
    is the number of times that pixel value appears in the segment. 
    """
    histDict = Dict.empty(key_type=numbaTypeForImageType, value_type=types.uint32)
    segDict = Dict.empty(key_type=segIdNumbaType, value_type=histDict._dict_type)
    return segDict


def createPagedRat():
    """
    Create the dictionary for the paged RAT. Each element is a page of
    the RAT, with entries for a range of segment IDs. The key is the 
    segment ID of the first entry in the page. 

    The returned dictionary is initially empty. 

    """
    pagedRat = Dict.empty(key_type=segIdNumbaType, 
        value_type=RatPage.class_type.instance_type)
    return pagedRat


@njit
def getRatPageId(segId):
    """
    For the given segment ID, return the page ID. This is the segment
    ID of the first segment in the page. 
    """
    pageId = (segId // RAT_PAGE_SIZE) * RAT_PAGE_SIZE
    return segIdNumbaType(pageId)

    
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
        raise PyShepSegTilingError(msg)
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
                if colType == STAT_DTYPE_INT:
                    colArr = ratPage.intcols[colArrayNdx]
                elif colType == STAT_DTYPE_FLOAT:
                    colArr = ratPage.floatcols[colArrayNdx]

                attrTbl.WriteArray(colArr, colNumber, start=startSegId)
            
            # Remove page after writing. 
            pagedRat.pop(pageId)


RAT_PAGE_SIZE = 100000
ratPageSpec = [
    ('startSegId', segIdNumbaType),
    ('intcols', numbaTypeForImageType[:,:]),
    ('floatcols', types.float32[:,:]),
    ('complete', types.boolean[:])
]

@jitclass(ratPageSpec)
class RatPage(object):
    """
    Hold a single page of the paged RAT
    """
    def __init__(self, numIntCols, numFloatCols, startSegId, numSeg):
        """
        Allocate arrays for int and float columns. Int columns are
        stored as signed int32, floats are float32. 
        
        startSegId is the segment ID number of the lowest segment in this page.
        numSeg is the number of segments within this page, normally the
        page size, but the last page will be smaller. 
        
        numIntCols and numFloatCols are as returned by makeFastStatSelection(). 
        
        """
        self.startSegId = startSegId
        self.intcols = numpy.empty((numIntCols, numSeg), dtype=numbaTypeForImageType)
        self.floatcols = numpy.empty((numFloatCols, numSeg), dtype=numpy.float32)
        self.complete = numpy.zeros(numSeg, dtype=types.boolean)
        if startSegId == shepseg.SEGNULLVAL:
            # The null segment is always complete
            self.complete[0] = True
            self.intcols[:, 0] = 0
            self.floatcols[:, 0] = 0
    
    def getIndexInPage(self, segId):
        """
        Return the index for the given segment, within the current
        page. 
        """
        return segId - self.startSegId

    def setRatVal(self, segId, colType, colArrayNdx, val):
        """
        Set the RAT entry for the given stat selection, for the given segment,
        to be the given value. 
        """
        ndxInPage = self.getIndexInPage(segId)
        if colType == STAT_DTYPE_INT:
            self.intcols[colArrayNdx, ndxInPage] = val
        elif colType == STAT_DTYPE_FLOAT:
            self.floatcols[colArrayNdx, ndxInPage] = val
    
    def setSegmentComplete(self, segId):
        """
        Flag that the given segment has had all stats calculated. 
        """
        ndxInPage = self.getIndexInPage(segId)
        self.complete[ndxInPage] = True
    
    def pageComplete(self):
        """
        Return True if the current page has been completed
        """
        return self.complete.all()
    

# Translate statistic name strings into integer ID values
STATID_MIN = 0
STATID_MAX = 1
STATID_MEAN = 2
STATID_STDDEV = 3
STATID_MEDIAN = 4
STATID_MODE = 5
STATID_PERCENTILE = 6
statIDdict = {
    'min':STATID_MIN,
    'max':STATID_MAX,
    'mean':STATID_MEAN,
    'stddev':STATID_STDDEV,
    'median':STATID_MEDIAN,
    'mode':STATID_MODE,
    'percentile':STATID_PERCENTILE
}
NOPARAM = -1

# Array indexes for the fast stat selection array
STATSEL_GLOBALCOLINDEX = 0
STATSEL_STATID = 1
STATSEL_COLTYPE = 2
STATSEL_COLARRAYINDEX = 3
STATSEL_PARAM = 4
STAT_DTYPE_INT = 0
STAT_DTYPE_FLOAT = 1

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
        
        statType = STAT_DTYPE_INT
        if statName in ('mean', 'stddev'):
            statType = STAT_DTYPE_FLOAT
        statsSelection_fast[i, STATSEL_COLTYPE] = statType
        
        if statType == STAT_DTYPE_INT:
            statsSelection_fast[i, STATSEL_COLARRAYINDEX] = intCount
            intCount += 1
        elif statType == STAT_DTYPE_FLOAT:
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
    keysArray = numpy.empty(size, dtype=numbaTypeForImageType)
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
segStatsSpec = [('pixVals', numbaTypeForImageType[:]), 
                ('counts', types.uint32[:]),
                ('pixCount', types.uint32),
                ('min', numbaTypeForImageType),
                ('max', numbaTypeForImageType),
                ('mean', types.float32),
                ('stddev', types.float32),
                ('median', numbaTypeForImageType),
                ('mode', numbaTypeForImageType)
               ]
@jitclass(segStatsSpec)
class SegmentStats(object):
    "Manage statistics for a single segment"
    def __init__(self, segmentHistDict):
        """
        Construct with generic statistics, given a typed 
        dictionary of the histogram counts of all values
        in the segment
        """
        self.pixVals, self.counts = getSortedKeysAndValuesForDict(segmentHistDict)
        # Total number of pixels in segment
        self.pixCount = self.counts.sum()

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
        countAtPcntile = self.pixCount * (percentile / 100)
        cumCount = 0
        i = 0
        while cumCount < countAtPcntile:
            cumCount += self.counts[i]
            i += 1
        pcntileVal = self.pixVals[i-1]
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
        return val


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
    very large images in a memory-efficient way. 
    
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
            
        colNum = attrTbl.GetColOfUsage(gdal.GFU_PixelCount)
        if colNum == -1:
            # Use GFT_Real to match rios.calcstats (And I think GDAL in general)
            attrTbl.CreateColumn('Histogram', gdal.GFT_Real, gdal.GFU_PixelCount)
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
