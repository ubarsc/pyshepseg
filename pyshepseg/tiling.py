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
:func:`pyshepseg.shepseg.doShepherdSegmentation` and
:func:`pyshepseg.shepseg.autoMaxSpectralDiff` for further details).

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

import os
import time
import shutil
import tempfile
import threading
import queue
from concurrent import futures
from multiprocessing import cpu_count

import numpy
from osgeo import gdal
import scipy.stats

from . import shepseg
from . import utils

gdal.UseExceptions()

DFLT_TEMPFILES_DRIVER = 'KEA'
DFLT_TEMPFILES_EXT = 'kea'

DFLT_TILESIZE = 4096
DFLT_OVERLAPSIZE = 1024

DFLT_CHUNKSIZE = 100000

TILESIZE = 1024

# Different concurrency types
CONC_NONE = "CONC_NONE"
CONC_THREADS = "CONC_THREADS"
CONC_FARGATE = "CONC_FARGATE"

# The two orientations of the overlap region
HORIZONTAL = 0
VERTICAL = 1
RIGHT_OVERLAP = 'right'
BOTTOM_OVERLAP = 'bottom'


class TiledSegmentationResult(object):
    """
    Result of tiled segmentation

    Attributes
    ----------
      maxSegId : shepseg.SegIdType
        Largest segment ID used in final segment image
      numTileRows : int
        Number of rows of tiles used
      numTileCols : int
        Number of columns of tiles used
      subsamplePcnt : float
        Percentage of image subsampled for clustering
      maxSpectralDiff : float
        The value used to limit segment merging (in all tiles)
      kmeans : sklearn.cluster.KMeans
        The sklearn KMeans object, after fitting
      hasEmptySegments : bool
        True if the segmentation contains segments with no pixels.
        This is an error condition, probably indicating that the
        merging of segments across tiles has produced inconsistent
        numbering. A warning message will also have been printed.
      outDs: gdal.Dataset
        Open GDAL dataset object to the output file. May not be set -
        see the returnGDALDS parameter to doTiledShepherdSegmentation.

    """
    def __init__(self):
        self.maxSegId = None
        self.numTileRows = None
        self.numTileCols = None
        self.subsamplePcnt = None
        self.maxSpectralDiff = None
        self.kmeans = None
        self.hasEmptySegments = None
        self.outDs = None


def fitSpectralClustersWholeFile(inDs, bandNumbers, numClusters=60, 
        subsamplePcnt=None, imgNullVal=None, 
        fixedKMeansInit=False):
    """
    Given an open raster Dataset, read a selected sample of pixels
    and use these to fit a spectral cluster model. Uses GDAL
    to read the pixels, and shepseg.fitSpectralClusters() 
    to do the fitting.

    Parameters
    ----------
      inDs : gdal.Dataset
        Open GDAL Dataset object for the input raster
      bandNumbers : list of int (or None)
        List of GDAL band numbers for the bands of interest. If
        None, then use all bands in the dataset. Note that GDAL band
        numbers start at 1.
      numClusters : int
        Desired number of clusters
      subsamplePcnt : float or None
        Percentage of pixels to use in fitting. If it is None, then
        a suitable subsample is calculated such that around one million
        pixels are sampled. (Note - this would include null pixels, so
        if the image is dominated by nulls, this would undersample.)
        No further subsampling is carried out by fitSpectralClusters().
      imgNullVal : float or None
        Pixels with this value in the input raster are ignored. If None,
        the NoDataValue from the raster file is used
      fixedKMeansInit : bool
        If True, then use a fixed estimate for the initial KMeans cluster
        centres. See shepseg.fitSpectralClusters() for details.

    Returns
    -------
      kmeansObj : sklearn.cluster.KMeans
        The fitted KMeans object
      subsamplePcnt : float
        The subsample percentage actually used
      imgNullVal : float
        The image null value (possibly read from the file)

    """
    if subsamplePcnt is None:
        # We will try to sample roughly this many pixels
        dfltTotalPixels = 1000000
        totalImagePixels = inDs.RasterXSize * inDs.RasterYSize
        # subsampleProp is the proportion of rows and columns sampled (hence
        # sqrt of total proportion).
        subsampleProp = numpy.sqrt(dfltTotalPixels / totalImagePixels)
        # Must be <= 1, i.e. sampling 100% of the data
        subsampleProp = min(1, subsampleProp)
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
    Return the null value for the given dataset

    Parameters
    ----------
      inDs : gdal.Dataset
        Open input Dataset
      bandNumbers : list of int
        GDAL band numbers of interest

    Returns
    -------
      imgNullVal : float or None
        Null value from input raster, None if there is no null value

    Raises
    ------
      PyShepSegTilingError
        If not all bands have the same null value

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

    Note that one can, in principle, do this directly using GDAL. 
    However, if overview layers are present in the file, it will use
    these, and so is dependent on how these were created. Since 
    these are often created just for display purposes, past experience
    has shown that they are not always to be trusted as data, 
    so we have chosen to always go directly to the full resolution 
    image. 

    Parameters
    ----------
      bandObj : gdal.Band
        An open Band object for input
      subsampleProp : float
        The proportion by which to sub-sample (i.e. a value between
        zero and 1, applied to rows and columns separately)

    Returns
    -------
      imgSub : <dtype> ndarray (nRowsSub, nColsSub)
        A numpy array of the image subsample, equivalent to
        calling gdal.Band.ReadAsArray()

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
        """
        Add a new tile to the set

        Parameters
        ----------
          xpos, ypos : int
            Pixel column & row of top left pixel of tile
          xsize, ysize : int
            Number of pixel columns & rows in tile
          col, row : int
            Tile column & row

        """
        self.tiles[(col, row)] = (xpos, ypos, xsize, ysize)
        
    def getNumTiles(self):
        """
        Get total number of tiles in the set

        Returns
        -------
          numTiles : int
            Total number of tiles
        """
        return len(self.tiles)
        
    def getTile(self, col, row):
        """
        Return the position and shape of the requested tile, as
        a single tuple of values

        Parameters
        ----------
          col, row : int
            Tile column & row

        Returns
        -------
          xpos, ypos : int
            Pixel column & row of top left pixel of tile
          xsize, ysize : int
            Number of pixel columns & rows in tile

        """

        return self.tiles[(col, row)]


def getTilesForFile(ds, tileSize, overlapSize):
    """
    Return a TileInfo object for a given file and input
    parameters.

    Parameters
    ----------
      ds : gdal.Dataset
        Open GDAL Dataset object for raster to be tiles
      tileSize : int
        Size of tiles, in pixels. Individual tiles may end up being
        larger in either direction, when they meet the edge of the raster,
        to ensure we do not use very small tiles
      overlapSize : int
        Number of pixels by which tiles will overlap

    Returns
    -------
      tileInfo : TileInfo
        TileInfo object detailing the sizes and positions of all tiles
        across the raster

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
        tempfilesCreationOptions=[], writeHistogram=True, returnGDALDS=False,
        concurrencyCfg=None):
    """
    Run the Shepherd segmentation algorithm in a memory-efficient
    manner, suitable for large raster files. Runs the segmentation
    on separate (overlapping) tiles across the raster, then stitches these
    together into a single output segment raster.

    The initial spectral clustering is performed on a sub-sample
    of the whole raster (using fitSpectralClustersWholeFile), 
    to create consistent clusters. These are then used as seeds 
    for all individual tiles. Note that subsamplePcnt is used at 
    this stage, over the whole raster, and is not passed through to 
    shepseg.doShepherdSegmentation() for any further sub-sampling.

    Most of the arguments are passed through to 
    shepseg.doShepherdSegmentation, and are described in the docstring
    for that function.

    Parameters
    ----------
      infile : str
        Filename of input raster
      outfile : str
        Filename of output segmentation raster
      tileSize : int
        Desired width & height (in pixels) of the tiles (i.e.
        desired tiles have shape (tileSize, tileSize). Tiles on the
        right and bottom edges of the input image may end up slightly
        larger than tileSize to ensure there are no small tiles.
      overlapSize : int
        Number of pixels to overlap tiles. The overlap area is a rectangle,
        this many pixels wide, which is covered by both adjacent tiles.
      minSegmentSize : int
        Minimum number of pixels in a segment
      numClusters : int
        Number of clusters to request in k-means clustering
      bandNumbers : list of int
        The GDAL band numbers (i.e. start at 1) of the bands of input raster
        to use for segmentation
      subsamplePcnt : float or None
        See fitSpectralClustersWholeFile()
      maxSpectralDiff : float or str
        See shepseg.doShepherdSegmentation()
      spectDistPcntile : int
        See shepseg.doShepherdSegmentation()
      imgNullVal : float or None
        If given, use this as the null value for the input raster. If None,
        use the value defined in the raster file
      fixedKMeansInit : bool
        If True, use a fixed set of initial cluster centres for the KMeans
        clustering. This is good to ensure exactly reproducible results
      fourConnected : bool
        If True, use 4-way connectedness, otherwise use 8-way
      verbose : bool
        If True, print informative messages during processing (to stdout)
      simpleTileRecode : bool
        If True, use only a simple tile recoding procedure. See
        stitchTiles() for more detail
      outputDriver : str
        The short name of the GDAL format driver to use for output file
      creationOptions : list of str
        The GDAL output creation options to match the outputDriver
      kmeansObj : sklearn.cluster.KMeans
        See shepseg.doShepherdSegmentation() for details
      tempfilesDriver : str
        Short name of GDAL driver to use for temporary raster files
      tempfilesExt : str
        File extension to use for temporary raster files
      tempfilesCreationOptions : list of str
        GDAL creation options to use for temporary raster files
      writeHistogram: bool
        Deprecated, and ignored. The histogram is always written.
      returnGDALDS: bool
        Whether to set the outDs member of TiledSegmentationResult
        when returning. If set, this will be open in update mode.
      concurrencyCfg: ConcurrencyConfig
        Configuration for segmentation concurrency. Default is None,
        meaning no concurrency.

    Returns
    -------
      tileSegResult : TiledSegmentationResult

    """
    if concurrencyCfg is None:
        concurrencyCfg = ConcurrencyConfig()

    concurrencyType = concurrencyCfg.concurrencyType
    concurrencyMgrClass = selectConcurrencyClass(concurrencyType)
    concurrencyMgr = concurrencyMgrClass(infile, outfile, tileSize,
        overlapSize, minSegmentSize, numClusters, bandNumbers, subsamplePcnt,
        maxSpectralDiff, imgNullVal, fixedKMeansInit, fourConnected, verbose,
        simpleTileRecode, outputDriver, creationOptions, spectDistPcntile,
        kmeansObj, tempfilesDriver, tempfilesCreationOptions, writeHistogram,
        returnGDALDS, concurrencyCfg)

    concurrencyMgr.initialize()
    concurrencyMgr.segmentAllTiles()
    concurrencyMgr.shutdown()

    tiledSegResult = TiledSegmentationResult()
    tiledSegResult.maxSegId = concurrencyMgr.maxSegId
    tiledSegResult.numTileRows = concurrencyMgr.tileInfo.nrows
    tiledSegResult.numTileCols = concurrencyMgr.tileInfo.ncols
    tiledSegResult.subsamplePcnt = concurrencyMgr.subsamplePcnt
    tiledSegResult.maxSpectralDiff = concurrencyMgr.maxSpectralDiff
    tiledSegResult.kmeans = concurrencyMgr.kmeansObj
    tiledSegResult.hasEmptySegments = concurrencyMgr.hasEmptySegments
    if returnGDALDS:
        tiledSegResult.outDs = concurrencyMgr.outDs
    
    return tiledSegResult


def selectConcurrencyClass(concurrencyType):
    """
    Choose the sub-class corresponding to the given concurrencyType
    """
    concMgrClass = None
    subclasses = SegmentationConcurrencyMgr.__subclasses__()
    for c in subclasses:
        if c.concurrencyType == concurrencyType:
            concMgrClass = c

    if concMgrClass is None:
        msg = f"Unknown concurrencyType '{concurrencyType}'"
        raise ValueError(msg)
    return concMgrClass


class ConcurrencyConfig:
    """
    Configuration for segmentation concurrency
    """
    def __init__(self, concurrencyType=CONC_NONE, numWorkers=0,
            maxConcurrentReads=20):
        self.concurrencyType = concurrencyType
        self.numWorkers = numWorkers
        self.maxConcurrentReads = maxConcurrentReads
    

class SegmentationConcurrencyMgr:
    """
    Base class for segmentation concurrency
    """
    concurrencyType = CONC_NONE

    def __init__(self, infile, outfile, tileSize, overlapSize, minSegmentSize,
            numClusters, bandNumbers, subsamplePcnt, maxSpectralDiff,
            imgNullVal, fixedKMeansInit, fourConnected, verbose,
            simpleTileRecode, outputDriver, creationOptions, spectDistPcntile,
            kmeansObj, tempfilesDriver, tempfilesCreationOptions,
            writeHistogram, returnGDALDS, concCfg):
        """
        Constructor. Just saves all its arguments to self, and does a couple
        of quick checks.
        """
        self.infile = infile
        self.outfile = outfile
        self.tileSize = tileSize
        self.overlapSize = overlapSize
        self.minSegmentSize = minSegmentSize
        self.numClusters = numClusters
        self.bandNumbers = bandNumbers
        self.subsamplePcnt = subsamplePcnt
        self.maxSpectralDiff = maxSpectralDiff
        self.imgNullVal = imgNullVal
        self.fixedKMeansInit = fixedKMeansInit
        self.fourConnected = fourConnected
        self.verbose = verbose
        self.simpleTileRecode = simpleTileRecode
        self.outputDriver = outputDriver
        self.creationOptions = creationOptions
        self.spectDistPcntile = spectDistPcntile
        self.kmeansObj = kmeansObj
        self.tempfilesDriver = tempfilesDriver
        self.tempfilesCreationOptions = tempfilesCreationOptions
        self.writeHistogram = writeHistogram
        self.returnGDALDS = returnGDALDS
        self.concurrencyCfg = concCfg
        if concCfg.numWorkers > 0:
            self.readSemaphore = threading.BoundedSemaphore(
                value=concCfg.maxConcurrentReads)

        if (self.overlapSize % 2) != 0:
            raise PyShepSegTilingError("Overlap size must be an even number")

        for driverName in [tempfilesDriver, outputDriver]:
            drvr = gdal.GetDriverByName(driverName)
            if drvr is None:
                msg = "This GDAL does not support driver '{}'".format(driverName)
                raise PyShepSegTilingError(msg)
            if driverName == tempfilesDriver:
                self.tempfilesExt = drvr.GetMetadataItem('DMD_EXTENSION')

        self.specificChecks()

    def specificChecks(self):
        """
        Checks which are specific to the subclass. Called at the
        end of __init__().
        """

    def initialize(self):
        """
        Runs initial phase of segmentation. This does not have any concurrency,
        so is the same for every concurrencyType. The main job is to do the
        spectral clustering, setting self.kmeansObj
        """
        if self.verbose:
            print("Starting tiled segmentation")

        inDs = gdal.Open(self.infile)

        if self.bandNumbers is None:
            self.bandNumbers = range(1, self.inDs.RasterCount + 1)

        t0 = time.time()
        if self.kmeansObj is None:
            fitReturn = fitSpectralClustersWholeFile(inDs, self.bandNumbers,
                    self.numClusters, self.subsamplePcnt, self.imgNullVal,
                    self.fixedKMeansInit)
            (self.kmeansObj, self.subsamplePcnt, self.imgNullVal) = fitReturn
                
            if self.verbose:
                print("KMeans of whole raster {:.2f} seconds".format(time.time() - t0))
                print("Subsample Percentage={:.2f}".format(self.subsamplePcnt))

        elif self.imgNullVal is None:
            # make sure we have the null value, even if they have supplied the kMeans
            self.imgNullVal = getImgNullValue(self.inDs, self.bandNumbers)

        self.tileInfo = getTilesForFile(inDs, self.tileSize, self.overlapSize)
        if self.verbose:
            print("Found {} tiles, with {} rows and {} cols".format(
                self.tileInfo.getNumTiles(), self.tileInfo.nrows, self.tileInfo.ncols))

        # Save some info on the input file to use for the output file
        self.inXsize = inDs.RasterXSize
        self.inYsize = inDs.RasterYSize
        self.inProj = inDs.GetProjection()
        self.inGeoTransform = inDs.GetGeoTransform()

    def shutdown(self):
        """
        Any explicit shutdown operations
        """

    def startQueues(self):
        """
        Start in and out queues, if required
        """

    def startWorkers(self):
        """
        Start segmentation workers, if required
        """

    def segmentAllTiles(self):
        """
        Runs segmentation for all tiles in the input image, and writes the output
        file.
        """

    def getTileSegmentation(self, col, row):
        """
        Get the requested tile of segmentation output
        """

    def saveCompletedTiles(self):
        """
        Save any completed tiles of segmentation output, if required. This
        is used by concurrent subclasses to grab completed tiles from
        the output queue, and hang onto them until we are ready to write them.
        """
        completedTile = self.popFromQue(self.outQue)
        while completedTile is not None:
            (col, row, segResult) = completedTile
            self.segResultCache[(col, row)] = segResult
            completedTile = self.popFromQue(self.outQue)

    @staticmethod
    def overlapCacheKey(col, row, edge):
        """
        Return the temporary cache key used for the overlap array

        Parameters
        ----------
          col, row : int
            Tile column & row numbers
          edge : {right', 'bottom'}
            Indicates from which edge of the given tile the overlap is taken

        Returns
        -------
          cachekey : str
            Identifying key for the overlap
        """
        cachekey = '{}_{}_{}'.format(edge, col, row)
        return cachekey

    def saveOverlap(self, overlapCacheKey, overlapData):
        """
        Save given overlap data in cache, under the given key. This may be
        in-memory or on-disk, depending on the sub-class
        """

    def loadOverlap(self, overlapCacheKey):
        """
        Load the requested overlap data from cache
        """

    def stitchTiles(self):
        """
        Recombine individual tiles into a single segment raster output 
        file. Segment ID values are recoded to be unique across the whole
        raster, and contiguous.

        Sets maxSegId and outDs on self.

        """
        marginSize = int(self.overlapSize / 2)

        outDrvr = gdal.GetDriverByName(self.outputDriver)

        if os.path.exists(self.outfile):
            outDrvr.Delete(self.outfile)

        outType = gdal.GDT_UInt32

        outDs = outDrvr.Create(self.outfile, self.inXsize, self.inYsize, 1, 
                    outType, self.creationOptions)
        outDs.SetProjection(self.inProj)
        outDs.SetGeoTransform(self.inGeoTransform)
        outBand = outDs.GetRasterBand(1)
        outBand.SetMetadataItem('LAYER_TYPE', 'thematic')
        outBand.SetNoDataValue(shepseg.SEGNULLVAL)

        tileInfoKeys = self.tileInfo.tiles.keys()
        colRowList = sorted(tileInfoKeys, key=lambda x: (x[1], x[0]))
        maxSegId = 0
        histAccum = HistogramAccumulator()

        if self.verbose:
            print("Stitching tiles together")
        reportedRow = -1
        i = 0
        # Currently this loop is a polling loop, checking on every iteration
        # whether the desired tile has been done. It should be more event-driven,
        # but am working up to that.
        while i < len(colRowList):
            (col, row) = colRowList[i]

            if self.verbose and row != reportedRow:
                print("Stitching tile row {}".format(row))
            reportedRow = row

            self.saveCompletedTiles()

            (xpos, ypos, xsize, ysize) = self.tileInfo.getTile(col, row)
            tileData = self.getTileSegmentation(col, row)

            if tileData is not None:
                top = marginSize
                bottom = ysize - marginSize
                left = marginSize
                right = xsize - marginSize

                xout = xpos + marginSize
                yout = ypos + marginSize

                rightName = self.overlapCacheKey(col, row, RIGHT_OVERLAP)
                bottomName = self.overlapCacheKey(col, row, BOTTOM_OVERLAP)

                if row == 0:
                    top = 0
                    yout = ypos

                if row == (self.tileInfo.nrows - 1):
                    bottom = ysize
                    bottomName = None

                if col == 0:
                    left = 0
                    xout = xpos

                if col == (self.tileInfo.ncols - 1):
                    right = xsize
                    rightName = None

                if self.simpleTileRecode:
                    nullmask = (tileData == shepseg.SEGNULLVAL)
                    tileData += maxSegId
                    tileData[nullmask] = shepseg.SEGNULLVAL
                else:
                    tileData = self.recodeTile(tileData, maxSegId, row, col, 
                                top, bottom, left, right)

                tileDataTrimmed = tileData[top:bottom, left:right]
                outBand.WriteArray(tileDataTrimmed, xout, yout)
                histAccum.doHistAccum(tileDataTrimmed)

                if rightName is not None:
                    self.saveOverlap(rightName, tileData[:, -self.overlapSize:])
                if bottomName is not None:
                    self.saveOverlap(bottomName, tileData[-self.overlapSize:, :])    

                tileMaxSegId = tileDataTrimmed.max()
                maxSegId = max(maxSegId, tileMaxSegId)
                i += 1

        self.writeHistogramToFile(outBand, histAccum)
        self.hasEmptySegments = (histAccum.hist[1:] == 0).any()
        utils.estimateStatsFromHisto(outBand, histAccum.hist)
        self.maxSegId = maxSegId
        self.outDs = outDs

    def recodeTile(self, tileData, maxSegId, tileRow, tileCol,
            top, bottom, left, right):
        """
        Adjust the segment ID numbers in the current tile, to make them
        globally unique (and contiguous) across the whole mosaic.

        Make use of the overlapping regions of tiles above and left,
        to identify shared segments, and recode those to segment IDs 
        from the adjacent tiles (i.e. we change the current tile, not 
        the adjacent ones). Non-shared segments are increased so they 
        are larger than previous values. 

        Parameters
        ----------
          tileData : shepseg.SegIdType ndarray (tileNrows, tileNcols)
            The array of segment IDs for a single image tile
          maxSegId : shepseg.SegIdType
            The current maximum segment ID for all preceding tiles.
          tileRow, tileCol : int
            The row/col numbers of this tile, within the whole-mosaic
            tile numbering scheme. (These are not pixel numbers, but tile
            grid numbers)
          top, bottom, left, right : int
            Pixel coordinates *within tile* of the non-overlap region of
            the tile.

        Returns
        -------
          newTileData : shepseg.SegIdType ndarray (tileNrows, tileNcols)
            A copy of tileData, with new segment ID numbers.

        """
        # The A overlaps are from the current tile. The B overlaps 
        # are the same regions from the earlier adjacent tiles, and
        # we load them here from the saved .npy files. 
        topOverlapA = tileData[:self.overlapSize, :]
        leftOverlapA = tileData[:, :self.overlapSize]

        recodeDict = {}    

        # Read in the bottom and right regions of the adjacent tiles
        if tileRow > 0:
            topOverlapCacheKey = self.overlapCacheKey(tileCol, tileRow - 1, 
                                    BOTTOM_OVERLAP)
            topOverlapB = self.loadOverlap(topOverlapCacheKey)

            self.recodeSharedSegments(tileData, topOverlapA, topOverlapB, 
                HORIZONTAL, recodeDict)

        if tileCol > 0:
            leftOverlapCacheKey = self.overlapCacheKey(tileCol - 1, tileRow, 
                                    RIGHT_OVERLAP)
            leftOverlapB = self.loadOverlap(leftOverlapCacheKey)

            self.recodeSharedSegments(tileData, leftOverlapA, leftOverlapB, 
                VERTICAL, recodeDict)

        (newTileData, newMaxSegId) = self.relabelSegments(tileData, recodeDict, 
            maxSegId, top, bottom, left, right)

        return newTileData

    @staticmethod
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

        overlapA and overlapB are numpy arrays of pixels in the overlap
        region in question, giving the segment ID numbers in the two tiles.
        The values in overlapB are from the earlier tile, and those in
        overlapA are from the current tile.

        It is critically important that the overlapping region is either
        at the top or the left of the current tile, as this means that 
        the row and column numbers of pixels in the overlap arrays 
        match the same pixels in the full tile. This cannot be used
        for overlaps on the right or bottom of the current tile.

        Parameters
        ----------
          tileData : shepseg.SegIdType ndarray (tileNrows, tileNcols)
            Tile subset of segment ID image
          overlapA, overlapB : shepseg.SegIdType ndarray (overlapNrows, overlapNcols)
            Tile overlap subsets of segment ID image
          orientation : {HORIZONTAL, VERTICAL}
            The orientation parameter defines whether we are dealing with
            overlap at the top (orientation == HORIZONTAL) or the left
            (orientation == VERTICAL).
          recodeDict : dict
            Keys and values are both segment ID numbers. Defines the mapping
            which recodes segment IDs. Updated in place.

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
            if __class__.crossesMidline(overlapA, segLoc[segid], orientation):
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
            # change in scipy 1.9.0. use keepdims=True for old behaviour
            # but that breaks older scipy. So workaround by seeing what returned
            if numpy.isscalar(modeObj.mode):
                segIdFromB = modeObj.mode
            else:
                segIdFromB = modeObj.mode[0]

            # Now record this recoding relationship
            recodeDict[segid] = segIdFromB

    @staticmethod
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

        Parameters
        ----------
          tileData : shepseg.SegIdType ndarray (tileNrows, tileNcols)
            Segment IDs of tile
          recodeDict : dict
            Keys and values are segment ID numbers. Defines mapping
            for segment relabelling
          maxSegId : shepseg.SegIdType
            Maximum segment ID number
          top, bottom, left, right : int
            Pixel coordinates *within tile* of the non-overlap region of
            the tile.

        Returns
        -------
            newTileData : shepseg.SegIdType ndarray (tileNrows, tileNcols)
              Segment IDs of tile, after relabelling
            newMaxSegId : shepseg.SegIdType
              New maximum segment ID after relabelling

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

    @staticmethod
    def crossesMidline(overlap, segLoc, orientation):
        """
        Check whether the given segment crosses the midline of the
        given overlap. If it does not, then it will lie entirely within
        exactly one tile, but if it does cross, then it will need to be
        re-coded across the midline.

        Parameters
        ----------
          overlap : shepseg.SegIdType ndarray (overlapNrows, overlapNcols)
            Array of segments just for this overlap region
          segLoc : shepseg.RowColArray
            The row/col coordinates (within the overlap array) of the
            pixels for the segment of interest
          orientation : {HORIZONTAL, VERTICAL}
            Indicates the orientation of the midline

        Returns
        -------
          crosses : bool
            True if the given segment crosses the midline

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

    @staticmethod
    def writeHistogramToFile(outBand, histAccum):
        """
        Write the accumulated histogram to the output segmentation file
        """
        attrTbl = outBand.GetDefaultRAT()
        numTableRows = len(histAccum.hist)
        if attrTbl.GetRowCount() != numTableRows:
            attrTbl.SetRowCount(numTableRows)
            
        colNum = attrTbl.GetColOfUsage(gdal.GFU_PixelCount)
        if colNum == -1:
            # Use GFT_Real to match rios.calcstats (And I think GDAL in general)
            attrTbl.CreateColumn('Histogram', gdal.GFT_Real, gdal.GFU_PixelCount)
            colNum = attrTbl.GetColumnCount() - 1
        attrTbl.WriteArray(histAccum.hist, colNum)


class SegNoConcurrencyMgr(SegmentationConcurrencyMgr):
    """
    Runs tiled segmentation with no concurrency
    """
    concurrencyType = CONC_NONE

    def segmentAllTiles(self):
        """
        Run segmentation for all tiles, and write output image. Just runs all
        tiles in sequence, and then the recode and stitch together for final
        output.
        """
        # create a temp directory for use in splitting out tiles, overlaps etc
        self.tempDir = tempfile.mkdtemp()
        # GDAL driver for temporary files
        outDrvr = gdal.GetDriverByName(self.tempfilesDriver)

        self.tileFilenames = {}
        inDs = gdal.Open(self.infile)

        tileInfoKeys = self.tileInfo.tiles.keys()
        colRowList = sorted(tileInfoKeys, key=lambda x: (x[1], x[0]))
        tileNum = 1
        for col, row in colRowList:
            if self.verbose:
                print("\nDoing tile {} of {}: row={}, col={}".format(
                    tileNum, len(colRowList), row, col))

            (xpos, ypos, xsize, ysize) = self.tileInfo.getTile(col, row)
            lyrDataList = []
            for bandNum in self.bandNumbers:
                lyr = inDs.GetRasterBand(bandNum)
                lyrData = lyr.ReadAsArray(xpos, ypos, xsize, ysize)
                lyrDataList.append(lyrData)

            img = numpy.array(lyrDataList)

            segResult = shepseg.doShepherdSegmentation(img, 
                        minSegmentSize=self.minSegmentSize,
                        maxSpectralDiff=self.maxSpectralDiff,
                        imgNullVal=self.imgNullVal, 
                        fourConnected=self.fourConnected,
                        kmeansObj=self.kmeansObj, 
                        verbose=self.verbose,
                        spectDistPcntile=self.spectDistPcntile)

            filename = 'tile_{}_{}.{}'.format(col, row, self.tempfilesExt)
            filename = os.path.join(self.tempDir, filename)
            self.writeTile(segResult, filename, outDrvr, xpos, ypos,
                xsize, ysize)
            self.tileFilenames[(col, row)] = filename

            tileNum += 1

        self.stitchTiles()

        shutil.rmtree(self.tempDir)

        # Save this in case it was deduced during segmentaion
        self.maxSpectralDiff = segResult.maxSpectralDiff

    def writeTile(self, segResult, filename, outDrvr, xpos, ypos,
            xsize, ysize):
        """
        Write the segmented tile to a temporary image file
        """
        if os.path.exists(filename):
            outDrvr.Delete(filename)

        outType = gdal.GDT_UInt32

        outDs = outDrvr.Create(filename, xsize, ysize, 1, outType, 
                    options=self.tempfilesCreationOptions)
        outDs.SetProjection(self.inProj)
        transform = self.inGeoTransform
        subsetTransform = list(transform)
        subsetTransform[0] = transform[0] + xpos * transform[1]
        subsetTransform[3] = transform[3] + ypos * transform[5]
        outDs.SetGeoTransform(tuple(subsetTransform))
        b = outDs.GetRasterBand(1)
        b.WriteArray(segResult.segimg)
        b.SetMetadataItem('LAYER_TYPE', 'thematic')
        b.SetNoDataValue(shepseg.SEGNULLVAL)

        del outDs

    def overlapCacheFilename(self, overlapCacheKey):
        """
        Return filename for given overlapCacheKey
        """
        return os.path.join(self.tempDir, f"{overlapCacheKey}.npy")

    def saveOverlap(self, overlapCacheKey, overlapData):
        """
        Save given overlap data to disk file
        """
        filename = self.overlapCacheFilename(overlapCacheKey)
        numpy.save(filename, overlapData)

    def loadOverlap(self, overlapCacheKey):
        """
        Load the requested overlap from disk cache
        """
        filename = self.overlapCacheFilename(overlapCacheKey)
        return numpy.load(filename)

    def saveCompletedTiles(self):
        """
        In this subclass, completed tiles are on disk, so do nothing
        """

    def getTileSegmentation(self, col, row):
        """
        Read the requested tile of segmentation output from disk
        """
        filename = self.tileFilenames[(col, row)]
        ds = gdal.Open(filename)
        tileData = ds.ReadAsArray()
        return tileData


class SegThreadsMgr(SegmentationConcurrencyMgr):
    """
    Run tiled segmentation with concurrency based on threads within the main
    process.
    """
    concurrencyType = CONC_THREADS
    segResultCache = {}
    overlapCache = {}

    def specificChecks(self):
        """
        Checks which are specific to the subclass
        """
        numCpus = cpu_count()
        if self.concurrencyCfg.numWorkers >= numCpus:
            msg = "numWorkers ({}) must be less than num CPUs ({})".format(
                self.concurrencyCfg.numWorkers, numCpus)
            raise PyShepSegTilingError(msg)

    def segmentAllTiles(self):
        """
        Run segmentation for all tiles, and write output image. Runs a number
        of worker threads, each working independently on individual tiles. The
        tiles to process are sent via a Queue, and the computed results are
        returned via a different Queue.

        Stitching the tiles together is run in the main thread, beginning as
        soon as the first tile is completed.
        """
        self.inQue = queue.Queue()
        self.outQue = queue.Queue()

        # Place all tiles in the inQue
        tileInfoKeys = self.tileInfo.tiles.keys()
        colRowList = sorted(tileInfoKeys, key=lambda x: (x[1], x[0]))
        for colRow in colRowList:
            self.inQue.put(colRow)

        self.startWorkers()
        self.stitchTiles()

    def startWorkers(self):
        """
        Start worker threads for segmenting tiles
        """
        # Start all the worker threads
        self.threadPool = futures.ThreadPoolExecutor(
            max_workers=self.concurrencyCfg.numWorkers)
        self.workerList = []
        for workerID in range(self.concurrencyCfg.numWorkers):
            worker = self.threadPool.submit(self.worker)
            self.workerList.append(worker)

    def worker(self):
        """
        Worker function. Called for each worker thread.
        """
        # Each worker needs its own open Dataset for the input file, as these
        # are not thread-safe
        inDs = gdal.Open(self.infile)

        colRow = self.popFromQue(self.inQue)
        while colRow is not None:
            (col, row) = colRow

            xpos, ypos, xsize, ysize = self.tileInfo.getTile(col, row)

            lyrDataList = []
            for bandNum in self.bandNumbers:
                with self.readSemaphore:
                    lyr = inDs.GetRasterBand(bandNum)
                    lyrData = lyr.ReadAsArray(xpos, ypos, xsize, ysize)
                    lyrDataList.append(lyrData)

            img = numpy.array(lyrDataList)

            segResult = shepseg.doShepherdSegmentation(img, 
                        minSegmentSize=self.minSegmentSize,
                        maxSpectralDiff=self.maxSpectralDiff,
                        imgNullVal=self.imgNullVal, 
                        fourConnected=self.fourConnected,
                        kmeansObj=self.kmeansObj, 
                        verbose=self.verbose,
                        spectDistPcntile=self.spectDistPcntile)

            self.outQue.put((col, row, segResult))
            colRow = self.popFromQue(self.inQue)

    @staticmethod
    def popFromQue(que):
        """
        Pop out the next item from the given Queue, returning None if
        the queue is empty.

        WARNING: don't use this if the queued items can be None
        """
        try:
            item = que.get(block=False)
        except queue.Empty:
            item = None
        return item

    def saveOverlap(self, overlapCacheKey, overlapData):
        """
        Save given overlap data to cache
        """
        self.overlapCache[overlapCacheKey] = overlapData

    def loadOverlap(self, overlapCacheKey):
        """
        Load the requested overlap from cache, and remove it from cache
        """
        return self.overlapCache.pop(overlapCacheKey)

    def getTileSegmentation(self, col, row):
        """
        Get the segmented tile output data from the local cache, and remove it
        from the cache
        """
        tileData = None
        segResult = self.segResultCache.get((col, row))
        if segResult is not None:
            self.segResultCache.pop((col, row))
            tileData = segResult.segimg
        return tileData


class HistogramAccumulator:
    """
    Accumulator for histogram for the output segmentation image. This
    allows us to accumulate the histogram incrementally, tile-by-tile.
    Note that there are simplifying assumptions about being uint32, and
    the null value being zero, so don't try to use this for anything else.
    """
    def __init__(self):
        self.hist = None

    def doHistAccum(self, arr):
        """
        Accumulate the histogram with counts from the given arr.
        """
        counts = numpy.bincount(arr.flatten())
        # Always remove counts for the null value
        counts[shepseg.SEGNULLVAL] = 0
        self.updateHist(counts)

    @staticmethod
    def addTwoHistograms(hist1, hist2):
        """
        Add the two given histograms together, and return the result.

        If one is longer than the other, the shorter one is added to it.

        """
        if hist1 is None:
            result = hist2
        else:
            l1 = len(hist1)
            l2 = len(hist2)
            if l1 > l2:
                hist1[:l2] += hist2
                result = hist1
            else:
                hist2[:l1] += hist1
                result = hist2
        return result

    def updateHist(self, newCounts):
        """
        Update the current histogram counts. If positive is True, then
        the counts for positive values are updated, otherwise those for the
        negative values are updated.

        """
        if len(newCounts) > 0:
            self.hist = self.addTwoHistograms(self.hist, newCounts)


class PyShepSegTilingError(Exception):
    pass
