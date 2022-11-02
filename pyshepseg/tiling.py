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
:func:`shepseg.doShepherdSegmentation` and :func:`shepseg.autoMaxSpectralDiff`
for further details). 

Because of these caveats, one should be very cautious about 
segmenting something like a continental-scale image. There is a 
lot of spectral variation across an area like a whole continent, 
and it may be unwise to use all the same parameters for the
whole area. 

Also included in this module is functionality for subsetting a large
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

import sys
import os
import time
import shutil
import tempfile

import numpy
from osgeo import gdal
import scipy.stats

from numba import njit
from numba.core import types
from numba.typed import Dict
from numba.experimental import jitclass

from . import shepseg

DFLT_TEMPFILES_DRIVER = 'KEA'
DFLT_TEMPFILES_EXT = 'kea'

DFLT_TILESIZE = 4096
DFLT_OVERLAPSIZE = 1024

DFLT_CHUNKSIZE = 100000

TILESIZE = 1024

# This type is used for all numba jit-ed data which is supposed to 
# match the data type of the imagery pixels. Int64 should be enough
# to hold any integer type, signed or unsigned, up to uint32. 
numbaTypeForImageType = types.int64
# This is the numba equivalent type of shepseg.SegIdType
segIdNumbaType = types.uint32


class TiledSegmentationResult(object):
    """
    Result of tiled segmentation
    
    Attributes:
      maxSegId: Largest segment ID used in final segment image
      numTileRows: Number of rows of tiles used
      numTileCols: Number of columns of tiles used
      subsamplePcnt: Percentage of image subsampled for clustering
      maxSpectralDiff: The value used to limit segment merging (in all tiles)
      kmeans: The sklearn KMeans object, after fitting
      hasEmptySegments: Boolean flag, this is an error condition
      
    """
    maxSegId = None
    numTileRows = None
    numTileCols = None
    subsamplePcnt = None
    maxSpectralDiff = None
    kmeans = None
    hasEmptySegments = None


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
        imgNullVal = getImgNullValue(inDs, bandNumbers)
    
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


def getImgNullValue(inDs, bandNumbers):
    """
    Return the null value for the given dataset. Raises an error if
    not all bands have the same null value. 
    """
    nullValArr = numpy.array([inDs.GetRasterBand(i).GetNoDataValue() 
        for i in bandNumbers])
    if (nullValArr != nullValArr[0]).any():
        raise PyShepSegTilingError("Different null values in some bands")
    imgNullVal = nullValArr[0]
    return imgNullVal


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
    skip = int(round(1. / subsampleProp))
    
    tileSize = TILESIZE
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
        if (ypos + ysize * 2) > ds.RasterYSize:
            ysize = ds.RasterYSize - ypos
            yDone = True
            if ysize == 0:
                break
    
        while not xDone:
            xsize = tileSize
            # ensure that we can fit another whole tile before the edge
            # - grow this tile if needed so we don't end up with slivers
            if (xpos + xsize * 2) > ds.RasterXSize:
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
        creationOptions=[], spectDistPcntile=50, kmeansObj=None,
        tempfilesDriver=DFLT_TEMPFILES_DRIVER, tempfilesExt=DFLT_TEMPFILES_EXT,
        tempfilesCreationOptions=[]):
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
    for the output file. creationOptions is the list of creation options
    to use for this driver.
    
    tempfilesDriver is a string of the name of the GDAL driver to use
    for temporary files. tempfilesExt the the extension to use and
    tempfilesCreationOptions is the list of creation options to use for 
    this driver.
    
    Most of the arguments are passed through to 
    shepseg.doShepherdSegmentation, and are described in the docstring 
    for that function. 
    
    Return an instance of TiledSegmentationResult class. 
    
    """
 
    inDs, bandNumbers, kmeansObj, subsamplePcnt, imgNullVal, tileInfo = (
        doTiledShepherdSegmentation_prepare(infile, tileSize, 
        overlapSize, bandNumbers, imgNullVal, kmeansObj, verbose, numClusters, 
        subsamplePcnt, fixedKMeansInit))
        
    # create a temp directory for use in splitting out tiles, overlaps etc
    tempDir = tempfile.mkdtemp()
    
    tileFilenames = {}

    colRowList = sorted(tileInfo.tiles.keys(), key=lambda x: (x[1], x[0]))
    tileNum = 1
    for col, row in colRowList:
        if verbose:
            print("\nDoing tile {} of {}: row={}, col={}".format(
                tileNum, len(colRowList), row, col))

        filename = 'tile_{}_{}.{}'.format(col, row, tempfilesExt)
        filename = os.path.join(tempDir, filename)

        segResult = doTiledShepherdSegmentation_doOne(inDs, filename, tileInfo, 
            col, row, bandNumbers, imgNullVal, kmeansObj, minSegmentSize, 
            maxSpectralDiff, verbose, spectDistPcntile, fourConnected,
            tempfilesDriver, tempfilesCreationOptions)
        tileFilenames[(col, row)] = filename

        tileNum += 1
        
    maxSegId, hasEmptySegments = doTiledShepherdSegmentation_finalize(inDs, 
        outfile, tileFilenames, tileInfo, overlapSize, tempDir, simpleTileRecode, 
        outputDriver, creationOptions, verbose)

    shutil.rmtree(tempDir)
    
    tiledSegResult = TiledSegmentationResult()
    tiledSegResult.maxSegId = maxSegId
    tiledSegResult.numTileRows = tileInfo.nrows
    tiledSegResult.numTileCols = tileInfo.ncols
    tiledSegResult.subsamplePcnt = subsamplePcnt
    tiledSegResult.maxSpectralDiff = segResult.maxSpectralDiff
    tiledSegResult.kmeans = kmeansObj
    tiledSegResult.hasEmptySegments = hasEmptySegments
    
    return tiledSegResult


def doTiledShepherdSegmentation_prepare(infile, tileSize=DFLT_TILESIZE, 
        overlapSize=DFLT_OVERLAPSIZE, bandNumbers=None, imgNullVal=None, 
        kmeansObj=None, verbose=False, numClusters=60, subsamplePcnt=None, 
        fixedKMeansInit=False):
    """
    Do all the preparation for the tiled segmentation. Call this first if 
    creating a parallel implementation, then call 
    doTiledShepherdSegmentation_doOne() for each tile in the returned TileInfo
    object.
    
    Returns a tuple with: (datasetObj, bandNumbers, kmeansObj, subsamplePcnt, 
    imgNullVal, tileInfo)
    """
        
    if verbose:
        print("Starting tiled segmentation")
    if (overlapSize % 2) != 0:
        raise PyShepSegTilingError("Overlap size must be an even number")

    inDs = gdal.Open(infile)

    if bandNumbers is None:
        bandNumbers = range(1, inDs.RasterCount + 1)

    t0 = time.time()
    if kmeansObj is None:
        kmeansObj, subsamplePcnt, imgNullVal = fitSpectralClustersWholeFile(inDs, 
            bandNumbers, numClusters, subsamplePcnt, imgNullVal, fixedKMeansInit)
        if verbose:
            print("KMeans of whole raster {:.2f} seconds".format(time.time() - t0))
            print("Subsample Percentage={:.2f}".format(subsamplePcnt))
            
    elif imgNullVal is None:
        # make sure we have the null value, even if they have supplied the kMeans
        imgNullVal = getImgNullValue(inDs, bandNumbers)
    
    tileInfo = getTilesForFile(inDs, tileSize, overlapSize)
    if verbose:
        print("Found {} tiles, with {} rows and {} cols".format(
            tileInfo.getNumTiles(), tileInfo.nrows, tileInfo.ncols))
            
    return inDs, bandNumbers, kmeansObj, subsamplePcnt, imgNullVal, tileInfo
    
    
def doTiledShepherdSegmentation_doOne(inDs, filename, tileInfo, col, row, 
        bandNumbers, imgNullVal, kmeansObj, minSegmentSize=50, 
        maxSpectralDiff='auto', verbose=False, spectDistPcntile=50, 
        fourConnected=True, tempfilesDriver=DFLT_TEMPFILES_DRIVER,
        tempfilesCreationOptions=[]):
    """
    Called from doTiledShepherdSegmentation(). Does a single tile and split
    out here as a seperate function so it can be called from in parallel 
    with other tiles if desired.
    
    tileInfo is object returned from doTiledShepherdSegmentation_prepare()
    and col, row describe the tile that this call will process.
    
    Return value is that from shepseg.doShepherdSegmentation().
    
    """

    outDrvr = gdal.GetDriverByName(tempfilesDriver)

    if outDrvr is None:
        msg = 'This GDAL does not support driver {}'.format(tempfilesDriver)
        raise SystemExit(msg)
    
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

    if os.path.exists(filename):
        outDrvr.Delete(filename)

    outType = gdal.GDT_UInt32

    outDs = outDrvr.Create(filename, xsize, ysize, 1, outType, 
                options=tempfilesCreationOptions)
    outDs.SetProjection(inDs.GetProjection())
    transform = inDs.GetGeoTransform()
    subsetTransform = list(transform)
    subsetTransform[0] = transform[0] + xpos * transform[1]
    subsetTransform[3] = transform[3] + ypos * transform[5]
    outDs.SetGeoTransform(tuple(subsetTransform))
    b = outDs.GetRasterBand(1)
    b.WriteArray(segResult.segimg)
    b.SetMetadataItem('LAYER_TYPE', 'thematic')
    b.SetNoDataValue(shepseg.SEGNULLVAL)

    del outDs
    
    return segResult


def doTiledShepherdSegmentation_finalize(inDs, outfile, tileFilenames, tileInfo, 
        overlapSize, tempDir, simpleTileRecode=False, outputDriver='KEA', 
        creationOptions=[], verbose=False):
    """
    Do the stitching of tiles and check for empty segments. Call after every 
    doTiledShepherdSegmentation_doOne() has completed for a given tiled
    segmentation.
    
    Returns a tuple with (axSegId, hasEmptySegments).
    
    """
        
    maxSegId = stitchTiles(inDs, outfile, tileFilenames, tileInfo, overlapSize,
        tempDir, simpleTileRecode, outputDriver, creationOptions, verbose)

    hasEmptySegments = checkForEmptySegments(outfile, maxSegId, overlapSize)
    
    return maxSegId, hasEmptySegments


def checkForEmptySegments(outfile, maxSegId, overlapSize):
    """
    Check the final segmentation for any empty segments. These
    can be problematic later, and should be avoided.
    """
    hist = calcHistogramTiled(outfile, maxSegId, writeToRat=False)
    emptySegIds = numpy.where(hist[1:] == 0)[0]
    numEmptySeg = len(emptySegIds)
    hasEmptySegments = (numEmptySeg > 0)
    if hasEmptySegments:
        msg = [
            "",
            "WARNING: Found {} segments with zero pixels".format(numEmptySeg),
            "    Segment IDs: {}".format(emptySegIds),
            "    This is caused by inconsistent joining of segmentation",
            "    tiles, and will probably cause trouble later on.",
            "    It is highly recommended to re-run with a larger overlap",
            "    size (currently {}), and if necessary a larger tile size".format(overlapSize),
            ""
        ]
        print('\n'.join(msg), file=sys.stderr)

    return hasEmptySegments


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
    
    colRows = sorted(tileInfo.tiles.keys(), key=lambda x: (x[1], x[0]))
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

        if row == (tileInfo.nrows - 1):
            bottom = ysize
            bottomName = None
            
        if col == 0:
            left = 0
            xout = xpos
            
        if col == (tileInfo.ncols - 1):
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
    # are the same regions from the earlier adjacent tiles, and
    # we load them here from the saved .npy files. 
    topOverlapA = tileData[:overlapSize, :]
    leftOverlapA = tileData[:, :overlapSize]
    
    recodeDict = {}    

    # Read in the bottom and right regions of the adjacent tiles
    if tileRow > 0:
        topOverlapFilename = overlapFilename(tileCol, tileRow - 1, 
                                BOTTOM_OVERLAP, tempDir)
        topOverlapB = numpy.load(topOverlapFilename)

        recodeSharedSegments(tileData, topOverlapA, topOverlapB, 
            HORIZONTAL, recodeDict)

    if tileCol > 0:
        leftOverlapFilename = overlapFilename(tileCol - 1, tileRow, 
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
    The values in overlapB are from the earlier tile, and those in 
    overlapA are from the current tile. 
    
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
    can be calculated directly using :func:`pyshepseg.shepseg.makeSegSize`.
    
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
    hist = numpy.zeros((maxSegId + 1), dtype=numpy.uint32)
    
    # Open the file
    if isinstance(segfile, gdal.Dataset):
        ds = segfile
    else:
        ds = gdal.Open(segfile, gdal.GA_Update)
    segband = ds.GetRasterBand(1)
    
    tileSize = TILESIZE
    (nlines, npix) = (segband.YSize, segband.XSize)
    numXtiles = int(numpy.ceil(npix / tileSize))
    numYtiles = int(numpy.ceil(nlines / tileSize))

    for tileRow in range(numYtiles):
        for tileCol in range(numXtiles):
            topLine = tileRow * tileSize
            leftPix = tileCol * tileSize
            xsize = min(tileSize, npix - leftPix)
            ysize = min(tileSize, nlines - topLine)
            
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


def subsetImage(inname, outname, tlx, tly, newXsize, newYsize, outformat, 
        creationOptions=[], origSegIdColName=None, maskImage=None):
    """
    Subset an image and "compress" the RAT so only values that 
    are in the new image are in the RAT. Note: the image values
    will be recoded in the process.
    
    gdal_translate seems to have a problem with files that
    have large RAT's so while that gets fixed do the subsetting
    in this function.
    
    tlx and tly are the top left of the image to extract in pixel coords
    of the input image. newXSize and newYSize are the size of the subset 
    to extract in pixels. 
    
    outformat is the GDAL driver name to use for the output image and
    creationOptions is a list of creation options to use for creating the
    output.
    
    If origSegIdColName is not None, a column of this name will be created
    in the output file that has the original segment ids so they can be
    linked back to the input file.
    
    If maskFile is not None then only pixels != 0 in this image are considered.
    It is assumed that the top left of this image is at tlx, tly on the 
    input image and its size is (newXsize, newYsize).
    
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
    recodeDict = Dict.empty(key_type=segIdNumbaType,
        value_type=segIdNumbaType)  # keyed on original ID - value is new row ID
    histogramDict = Dict.empty(key_type=segIdNumbaType,
        value_type=segIdNumbaType)  # keyed on new ID - value is count
 
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
    tileSize = TILESIZE
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
    outPagedRat = createPagedRat()
    for startSegId in range(minInVal, maxInVal, RAT_PAGE_SIZE):
        # looping through in RAT_PAGE_SIZE pages
        endSegId = min(startSegId + RAT_PAGE_SIZE - 1, maxInVal)
        
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
    
    """
    numIntCols = inPage.intcols.shape[0]
    numFloatCols = inPage.floatcols.shape[0]
    maxSegId = len(recodeDict)
    for inRowInPage in range(inPage.intcols.shape[1]):
        inRow = segIdNumbaType(inPage.startSegId + inRowInPage)
        if inRow not in recodeDict:
            # this one is not in this subset, skip
            continue
        outRow = recodeDict[inRow]
        
        outPageId = getRatPageId(outRow)
        outRowInPage = outRow - outPageId
        if outPageId not in outPagedRat:
            numSegThisPage = min(RAT_PAGE_SIZE, (maxSegId - outPageId + 1))
            outPagedRat[outPageId] = RatPage(numIntCols, numFloatCols, 
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
    page = RatPage(numIntCols, numFloatCols, minVal, nrows)

    intColIdx = 0
    floatColIdx = 0
    for col in range(rat.GetColumnCount()):
        dtype = rat.GetTypeOfCol(col)
        data = rat.ReadAsArray(col, start=minVal, length=nrows)
        if dtype == gdal.GFT_Integer:
            readColDataIntoPage(page, data, intColIdx, STAT_DTYPE_INT, minVal)
            intColIdx += 1
        else:
            readColDataIntoPage(page, data, floatColIdx, STAT_DTYPE_FLOAT, minVal)
            floatColIdx += 1
    
    return page 


def copyColumns(inRat, outRat):
    """
    Copies columns between RATs. Note that empty
    columns will be created - no data is copied.
    
    Also return numIntCols and NumFloatCols for RatPage()
    processing.
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
    
    if maskData is not None then only pixels != 0 are considered. Assumed 
    that maskData is for the same area/size as tile.
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
                recodeDict[segId] = segIdNumbaType(outSegId)
            
            # write this new value to the output image    
            newval = recodeDict[segId]
            outData[y, x] = newval
            # update histogram
            if newval not in histogramDict:
                histogramDict[newval] = segIdNumbaType(0)
            histogramDict[newval] = segIdNumbaType(histogramDict[newval] + 1)
            
    return outData


def writeCompletedPagesForSubset(inRAT, outRAT, outPagedRat):
    """
    For the subset operation. Writes out any completed pages to outRAT
    using the inRAT as a template.
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


STAT_DTYPE_INT = 0
STAT_DTYPE_FLOAT = 1
    
RAT_PAGE_SIZE = 100000
ratPageSpec = [
    ('startSegId', segIdNumbaType),
    ('intcols', numbaTypeForImageType[:, :]),
    ('floatcols', types.float32[:, :]),
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
        Set the RAT entry for the given segment,
        to be the given value. 
        """
        ndxInPage = self.getIndexInPage(segId)
        if colType == STAT_DTYPE_INT:
            self.intcols[colArrayNdx, ndxInPage] = val
        elif colType == STAT_DTYPE_FLOAT:
            self.floatcols[colArrayNdx, ndxInPage] = val
            
    def getRatVal(self, segId, colType, colArrayNdx):
        """
        Get the RAT entry for the given segment.
        """
        ndxInPage = self.getIndexInPage(segId)
        if colType == STAT_DTYPE_INT:
            val = self.intcols[colArrayNdx, ndxInPage]
        elif colType == STAT_DTYPE_FLOAT:
            val = self.floatcols[colArrayNdx, ndxInPage]
        return val
    
    def setSegmentComplete(self, segId):
        """
        Flag that the given segment has had all stats calculated. 
        """
        ndxInPage = self.getIndexInPage(segId)
        self.complete[ndxInPage] = True
        
    def getSegmentComplete(self, segId):
        """
        Returns True if the segment has been flagged as complete
        """
        ndxInPage = self.getIndexInPage(segId)
        return self.complete[ndxInPage]
    
    def pageComplete(self):
        """
        Return True if the current page has been completed
        """
        return self.complete.all()


class PyShepSegTilingError(Exception):
    pass


class PyShepSegSubsetError(PyShepSegTilingError):
    pass
