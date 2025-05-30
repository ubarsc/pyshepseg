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

import sys
import os
import time
import shutil
import tempfile
import threading
import queue
import socket
import secrets
import random
from concurrent import futures
from multiprocessing import cpu_count
import multiprocessing.managers
import subprocess

import numpy
from osgeo import gdal, gdal_array
import scipy.stats
from numba import njit
from numba.core import types

from . import shepseg
from . import utils
from . import timinghooks
try:
    import boto3
except ImportError:
    boto3 = None


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
CONC_SUBPROC = "CONC_SUBPROC"

# The two orientations of the overlap region
HORIZONTAL = 0
VERTICAL = 1
RIGHT_OVERLAP = 'right'
BOTTOM_OVERLAP = 'bottom'

# This is the numba equivalent type of shepseg.SegIdType. No longer
# used here, but still (potentially) referenced from other modules.
segIdNumbaType = types.uint32


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
      timings : pyshepseg.timinghooks.Timers
        Timings for various key parts of the segmentation process
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
        self.timings = None


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
      writeHistogram : bool
        Deprecated, and ignored. The histogram is always written.
      returnGDALDS : bool
        Whether to set the outDs member of TiledSegmentationResult
        when returning. If set, this will be open in update mode.
      concurrencyCfg : SegmentationConcurrencyConfig
        Configuration for segmentation concurrency. Default is None,
        meaning no concurrency.

    Returns
    -------
      tileSegResult : TiledSegmentationResult

    """
    if concurrencyCfg is None:
        concurrencyCfg = SegmentationConcurrencyConfig()

    concurrencyType = concurrencyCfg.concurrencyType
    concurrencyMgrClass = selectConcurrencyClass(concurrencyType,
        SegmentationConcurrencyMgr)
    concurrencyMgr = concurrencyMgrClass(infile, outfile, tileSize,
        overlapSize, minSegmentSize, numClusters, bandNumbers, subsamplePcnt,
        maxSpectralDiff, imgNullVal, fixedKMeansInit, fourConnected, verbose,
        simpleTileRecode, outputDriver, creationOptions, spectDistPcntile,
        kmeansObj, tempfilesDriver, tempfilesCreationOptions, writeHistogram,
        returnGDALDS, concurrencyCfg)

    with concurrencyMgr.timings.interval('walltime'):
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
    tiledSegResult.timings = concurrencyMgr.timings
    if returnGDALDS:
        tiledSegResult.outDs = concurrencyMgr.outDs
    
    return tiledSegResult


def selectConcurrencyClass(concurrencyType, baseClass):
    """
    Choose the sub-class corresponding to the given concurrencyType
    """
    concMgrClass = None
    subclasses = baseClass.__subclasses__()
    for c in subclasses:
        if c.concurrencyType == concurrencyType:
            concMgrClass = c

    if concMgrClass is None:
        msg = f"Unknown concurrencyType '{concurrencyType}'"
        raise ValueError(msg)
    return concMgrClass


class SegmentationConcurrencyConfig:
    """
    Configuration for concurrency. This class can be used independantly to
    configure concurrency in either segmentation or per-segment statistics.
    """
    def __init__(self, concurrencyType=CONC_NONE, numWorkers=0,
            maxConcurrentReads=20, tileCompletionTimeout=60,
            fargateCfg=None):
        """
        Configuration for managing segmentation concurrency.

        Parameters
        ----------
          concurrencyType : One of {CONC_NONE, CONC_THREADS, CONC_FARGATE}
            The mechanism used for concurrency
          numWorkers : int
            Number of segmentation workers
          maxConcurrentReads : int
            Maximum number of concurrent reads. Each segmentation worker
            does its own reading of input data. Since the number of workers
            can be quite large, this could load the read device too heavily.
            Given that the read step is a very small component of each
            worker's activity, we can limit the number of concurrent reads
            to this value, without degrading throughput.
          tileCompletionTimeout : int
            Timeout (seconds) to wait for completion of each segmentation tile
          fargateCfg : None or instance of FargateConfig
            Configuration for AWS Fargate (when using CONC_FARGATE)

        """
        self.concurrencyType = concurrencyType
        self.numWorkers = numWorkers
        self.maxConcurrentReads = maxConcurrentReads
        self.tileCompletionTimeout = tileCompletionTimeout
        self.fargateCfg = fargateCfg
        if concurrencyType == CONC_FARGATE and fargateCfg is None:
            msg = "fargateCfg is required with CONC_FARGATE"
            raise PyShepSegTilingError(msg)
        if concurrencyType != CONC_FARGATE and fargateCfg is not None:
            msg = "fargateCfg is only used with CONC_FARGATE"
            raise PyShepSegTilingError(msg)


class FargateConfig:
    """
    Configuration for AWS Fargate
    """
    def __init__(self, containerImage=None, taskRoleArn=None,
            executionRoleArn=None, subnets=None,
            securityGroups=None, cpu='0.5 vCPU', memory='1GB',
            cpuArchitecture=None):
        """
        AWS Fargate configuration information. For use only with CONC_FARGATE.

        Parameters
        ----------
          containerImage : str
            URI of the container image to use for segmentation workers. This
            container must have pyshepseg installed. It can be the same
            container as used for the main script, as the entry point is
            over-written.
          taskRoleArn : str
            ARN for an AWS role. This allows your code to use AWS services.
            This role should include policies such as AmazonS3FullAccess,
            covering any AWS services the segmentation workers will need.
          executionRoleArn : str
            ARN for an AWS role. This allows ECS to use AWS services on
            your behalf. A good start is a role including
            AmazonECSTaskExecutionRolePolicy
          subnets : list of str
            List of subnet ID strings associated with the VPC in which
            workers will run.
          securityGroups : list of str
            Fargate. List of security group IDs associated with the VPC.
          cpu : str
            Number of CPU units requested for each segmentation worker,
            expressed in AWS's own units. For example, '0.5 vCPU', or
            '1024' (which corresponds to the same thing). Both must be strings.
            This helps Fargate to select a suitable VM instance type.
          memory : str
            Amount of memory requested for each segmentation worker,
            expressed in MiB, or with a units suffix. For example, '1024'
            or its equivalent '1GB'. This helps Fargate to select a suitable
            VM instance type.
          cpuArchitecture : str
            If given, selects the CPU architecture of the hosts to run
            worker on. Can be 'ARM64', defaults to 'X86_64'.

        """
        self.containerImage = containerImage
        self.taskRoleArn = taskRoleArn
        self.executionRoleArn = executionRoleArn
        self.subnets = subnets
        self.securityGroups = securityGroups
        self.cpu = cpu
        self.memory = memory
        self.cpuArchitecture = cpuArchitecture
    

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
        self.overlapCache = {}
        self.timings = timinghooks.Timers()

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
            self.bandNumbers = range(1, inDs.RasterCount + 1)

        t0 = time.time()
        if self.kmeansObj is None:
            with self.timings.interval('spectralclusters'):
                fitReturn = fitSpectralClustersWholeFile(inDs, self.bandNumbers,
                        self.numClusters, self.subsamplePcnt, self.imgNullVal,
                        self.fixedKMeansInit)
            (self.kmeansObj, self.subsamplePcnt, self.imgNullVal) = fitReturn
                
            if self.verbose:
                print("KMeans of whole raster {:.2f} seconds".format(time.time() - t0))
                print("Subsample Percentage={:.2f}".format(self.subsamplePcnt))

        elif self.imgNullVal is None:
            # make sure we have the null value, even if they have supplied the kMeans
            self.imgNullVal = getImgNullValue(inDs, self.bandNumbers)

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

    def setupNetworkComms(self):
        """
        Set up a NetworkDataChannel to communicate with the workers
        outside the main process (e.g. Fargate instances)
        """
        # The segDataDict is all the stuff which is just data, and can be
        # pickled and sent across to the workers.
        segDataDict = {}
        segDataDict['infile'] = self.infile
        segDataDict['tileInfo'] = self.tileInfo
        segDataDict['minSegmentSize'] = self.minSegmentSize
        segDataDict['maxSpectralDiff'] = self.maxSpectralDiff
        segDataDict['imgNullVal'] = self.imgNullVal
        segDataDict['fourConnected'] = self.fourConnected
        segDataDict['kmeansObj'] = self.kmeansObj
        segDataDict['verbose'] = self.verbose
        segDataDict['spectDistPcntile'] = self.spectDistPcntile
        segDataDict['bandNumbers'] = self.bandNumbers

        self.dataChan = NetworkDataChannel(inQue=self.inQue,
            segResultCache=self.segResultCache,
            forceExit=self.forceExit,
            exceptionQue=self.exceptionQue,
            segDataDict=segDataDict,
            readSemaphore=self.readSemaphore,
            timings=self.timings)

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
        segResult = self.segResultCache.waitForTile(col, row)
        if segResult is not None:
            tileData = segResult.segimg
        else:
            tileData = None
        return tileData

    def startWorkers(self):
        """
        Start segmentation workers, if required
        """

    def segmentAllTiles(self):
        """
        Run segmentation for all tiles, and write output image. Runs a number
        of segmentation workers, each working independently on individual
        tiles. The tiles to process are sent via a Queue, and the computed
        results are returned via a different Queue.

        Stitching the tiles together is run in the main thread, beginning as
        soon as the first tile is completed.
        """
        tileInfoKeys = self.tileInfo.tiles.keys()
        colRowList = sorted(tileInfoKeys, key=lambda x: (x[1], x[0]))

        self.inQue = queue.Queue()
        self.segResultCache = SegmentationResultCache(colRowList,
            timeout=self.concurrencyCfg.tileCompletionTimeout)
        self.forceExit = threading.Event()
        self.exceptionQue = queue.Queue()

        try:
            self.setupNetworkComms()

            # Place all tiles in the inQue
            for colRow in colRowList:
                self.inQue.put(colRow)

            self.startWorkers()
            with self.timings.interval('stitchtiles'):
                self.stitchTiles()
        finally:
            if hasattr(self, 'dataChan'):
                self.dataChan.shutdown()

    def checkWorkerExceptions(self):
        """
        Check if any workers raised exceptions. If so, raise a local exception
        with the WorkerErrorRecord.
        """
        # Check for worker exceptions
        if self.exceptionQue.qsize() > 0:
            exceptionRecord = self.exceptionQue.get()
            utils.reportWorkerException(exceptionRecord)
            msg = "The preceding exception was raised in a worker"
            raise PyShepSegTilingError(msg)

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

    def stitchTiles(self):
        """
        Recombine individual tiles into a single segment raster output 
        file. Segment ID values are recoded to be unique across the whole
        raster, and contiguous.

        Sets maxSegId and outDs on self.

        """
        marginSize = int(self.overlapSize / 2)

        if os.path.exists(self.outfile):
            drvr = gdal.IdentifyDriver(self.outfile)
            drvr.Delete(self.outfile)

        outType = gdal_array.NumericTypeCodeToGDALTypeCode(shepseg.SegIdType)

        outDrvr = gdal.GetDriverByName(self.outputDriver)
        outDs = outDrvr.Create(self.outfile, self.inXsize, self.inYsize, 1, 
                    outType, self.creationOptions)
        outDs.SetProjection(self.inProj)
        outDs.SetGeoTransform(self.inGeoTransform)
        self.setupOverviews(outDs)
        outBand = outDs.GetRasterBand(1)
        outBand.SetMetadataItem('LAYER_TYPE', 'thematic')
        outBand.SetNoDataValue(shepseg.SEGNULLVAL)

        tileInfoKeys = self.tileInfo.tiles.keys()
        colRowList = sorted(tileInfoKeys, key=lambda x: (x[1], x[0]))
        maxSegId = 0
        histAccum = HistogramAccumulator()

        workerError = False
        if self.verbose:
            print("Stitching tiles together")
        reportedRow = -1
        i = 0
        while i < len(colRowList) and not workerError:
            self.checkWorkerExceptions()

            (col, row) = colRowList[i]

            if self.verbose and row != reportedRow:
                print("Stitching tile row {}".format(row))
            reportedRow = row

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
                self.writeOverviews(outBand, tileDataTrimmed, xout, yout)
                histAccum.doHistAccum(tileDataTrimmed)

                if rightName is not None:
                    self.saveOverlap(rightName, tileData[:, -self.overlapSize:])
                if bottomName is not None:
                    self.saveOverlap(bottomName, tileData[-self.overlapSize:, :])

                tileMaxSegId = tileDataTrimmed.max()
                maxSegId = max(maxSegId, tileMaxSegId)
                i += 1
            else:
                workerError = True

        if not workerError:
            self.writeHistogramToFile(outBand, histAccum)
            self.hasEmptySegments = self.checkForEmptySegments(histAccum.hist,
                self.overlapSize)
            utils.estimateStatsFromHisto(outBand, histAccum.hist)
            self.maxSegId = maxSegId
            outDs.FlushCache()
            if self.returnGDALDS:
                self.outDs = outDs
            else:
                del outDs
        else:
            self.checkWorkerExceptions()

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

    def checkForEmptySegments(self, hist, overlapSize):
        """
        Check the final segmentation for any empty segments. These
        can be problematic later, and should be avoided. Prints a
        warning message if empty segments are found.

        Parameters
        ----------
          hist : ndarray of uint32
            Histogram counts for the segmentation raster
          overlapSize : int
            Number of pixels to use in overlaps between tiles

        Returns
        -------
          hasEmptySegments : bool
            True if there are segment ID numbers with no pixels

        """
        emptySegIds = numpy.where(hist[1:] == 0)[0] + 1
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

    def writeOverviews(self, outBand, arr, xOff, yOff):
        """
        Calculate and write out the overview layers for the tile
        given as arr.

        """
        nOverviews = len(self.overviewLevels)

        for j in range(nOverviews):
            band_ov = outBand.GetOverview(j)
            lvl = self.overviewLevels[j]
            # Offset from top-left edge
            o = lvl // 2
            # Sub-sample by taking every lvl-th pixel in each direction
            arr_sub = arr[o::lvl, o::lvl]
            # The xOff/yOff of the block within the sub-sampled raster
            xOff_sub = xOff // lvl
            yOff_sub = yOff // lvl
            # The actual number of rows and cols to write, ensuring we
            # do not go off the edges
            nc = band_ov.XSize - xOff_sub
            nr = band_ov.YSize - yOff_sub
            arr_sub = arr_sub[:nr, :nc]
            band_ov.WriteArray(arr_sub, xOff_sub, yOff_sub)

    def setupOverviews(self, outDs):
        """
        Calculate a suitable set of overview levels to use for output
        segmentation file, and set these up on the given Dataset. Stores
        the overview levels list as self.overviewLevels
        """
        # Work out a list of overview levels, starting with 4, until the
        # raster size (in largest direction) is smaller then finalOutSize.
        outSize = max(self.inXsize, self.inYsize)
        finalOutSize = 1024
        self.overviewLevels = []
        i = 2
        totalSizeOK = (outSize // (2 ** i)) >= finalOutSize
        while (totalSizeOK):
            self.overviewLevels.append(2 ** i)
            totalSizeOK = (outSize // (2 ** i)) >= finalOutSize
            i += 1

        # Create these overview layers on the dataset
        outDs.BuildOverviews("NEAREST", self.overviewLevels)


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
            with self.timings.interval('reading'):
                lyrDataList = []
                for bandNum in self.bandNumbers:
                    lyr = inDs.GetRasterBand(bandNum)
                    lyrData = lyr.ReadAsArray(xpos, ypos, xsize, ysize)
                    lyrDataList.append(lyrData)

            img = numpy.array(lyrDataList)

            with self.timings.interval('segmentation'):
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
            self.writeTileToTemp(segResult, filename, outDrvr, xpos, ypos,
                xsize, ysize)
            self.tileFilenames[(col, row)] = filename

            tileNum += 1

        with self.timings.interval('stitchtiles'):
            self.stitchTiles()

        shutil.rmtree(self.tempDir)

        # Save this in case it was deduced during segmentaion
        self.maxSpectralDiff = segResult.maxSpectralDiff

    def writeTileToTemp(self, segResult, filename, outDrvr, xpos, ypos,
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

    def getTileSegmentation(self, col, row):
        """
        Read the requested tile of segmentation output from disk
        """
        filename = self.tileFilenames[(col, row)]
        ds = gdal.Open(filename)
        tileData = ds.ReadAsArray()
        return tileData

    def checkWorkerExceptions(self):
        """
        Dummy. No workers, so no worker exceptions.
        """


class SegThreadsMgr(SegmentationConcurrencyMgr):
    """
    Run tiled segmentation with concurrency based on threads within the main
    process.
    """
    concurrencyType = CONC_THREADS

    def specificChecks(self):
        """
        Checks which are specific to the subclass
        """
        numCpus = cpu_count()
        if self.concurrencyCfg.numWorkers >= numCpus:
            msg = "numWorkers ({}) must be less than num CPUs ({})".format(
                self.concurrencyCfg.numWorkers, numCpus)
            raise PyShepSegTilingError(msg)

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
        try:
            # Each worker needs its own open Dataset for the input file, as these
            # are not thread-safe
            inDs = gdal.Open(self.infile)

            colRow = self.popFromQue(self.inQue)
            while colRow is not None and not self.forceExit.is_set():
                (col, row) = colRow

                xpos, ypos, xsize, ysize = self.tileInfo.getTile(col, row)

                with self.timings.interval('reading'):
                    lyrDataList = []
                    for bandNum in self.bandNumbers:
                        with self.readSemaphore:
                            lyr = inDs.GetRasterBand(bandNum)
                            lyrData = lyr.ReadAsArray(xpos, ypos, xsize, ysize)
                            lyrDataList.append(lyrData)

                img = numpy.array(lyrDataList)

                with self.timings.interval('segmentation'):
                    segResult = shepseg.doShepherdSegmentation(img,
                                minSegmentSize=self.minSegmentSize,
                                maxSpectralDiff=self.maxSpectralDiff,
                                imgNullVal=self.imgNullVal,
                                fourConnected=self.fourConnected,
                                kmeansObj=self.kmeansObj,
                                verbose=self.verbose,
                                spectDistPcntile=self.spectDistPcntile)

                self.segResultCache.addResult(col, row, segResult)
                colRow = self.popFromQue(self.inQue)
        except Exception as e:
            # Send a printable version of the exception back to main thread
            workerErr = utils.WorkerErrorRecord(e, 'segmentation')
            self.exceptionQue.put(workerErr)

    def shutdown(self):
        """
        Shut down the thread pool
        """
        self.forceExit.set()
        futures.wait(self.workerList)
        self.threadPool.shutdown()

    def setupNetworkComms(self):
        """
        Dummy. No network communications required.
        """


class SegFargateMgr(SegmentationConcurrencyMgr):
    """
    Run tiled segmentation with concurrency based on AWS Fargate workers.
    """
    concurrencyType = CONC_FARGATE

    def specificChecks(self):
        """
        Initial checks which are specific to the subclass
        """
        if boto3 is None:
            msg = "CONC_FARGATE requires boto3 to be installed"
            raise PyShepSegTilingError(msg)

    def startWorkers(self):
        """
        Start all segmentation workers as AWS Fargate tasks
        """
        concCfg = self.concurrencyCfg
        fargateCfg = concCfg.fargateCfg

        ecsClient = boto3.client("ecs")
        self.ecsClient = ecsClient

        jobIDstr = random.randbytes(4).hex()
        containerName = f'pyshepseg_{jobIDstr}_container'
        workerCmd = 'pyshepseg_segmentationworkercmd'
        containerDefs = [{'name': containerName,
                          'image': fargateCfg.containerImage,
                          'entryPoint': ['/usr/bin/env', workerCmd]}]

        # Create a private cluster
        self.clusterName = f'pyshepseg_{jobIDstr}_cluster'
        self.ecsClient.create_cluster(clusterName=self.clusterName)

        networkConf = {
            'awsvpcConfiguration': {
                'assignPublicIp': 'DISABLED',
                'subnets': fargateCfg.subnets,
                'securityGroups': fargateCfg.securityGroups
            }
        }

        taskFamily = f"pyshepseg_{jobIDstr}_task"
        taskDefParams = {
            'family': taskFamily,
            'networkMode': 'awsvpc',
            'requiresCompatibilities': ['FARGATE'],
            'containerDefinitions': containerDefs,
            'cpu': fargateCfg.cpu,
            'memory': fargateCfg.memory
        }
        if fargateCfg.taskRoleArn is not None:
            taskDefParams['taskRoleArn'] = fargateCfg.taskRoleArn
        if fargateCfg.executionRoleArn is not None:
            taskDefParams['executionRoleArn'] = fargateCfg.executionRoleArn
        if fargateCfg.cpuArchitecture is not None:
            taskDefParams['runtimePlatform'] = {'cpuArchitecture':
                fargateCfg.cpuArchitecture}

        taskDefResponse = self.ecsClient.register_task_definition(**taskDefParams)
        self.taskDefArn = taskDefResponse['taskDefinition']['taskDefinitionArn']

        runTaskParams = {
            'launchType': 'FARGATE',
            'cluster': self.clusterName,
            'networkConfiguration': networkConf,
            'taskDefinition': self.taskDefArn,
            'overrides': {'containerOverrides': [{
                "command": 'Dummy, to be over-written',
                'name': containerName}]}
        }

        channAddr = self.dataChan.addressStr()
        ctrOverrides = runTaskParams['overrides']['containerOverrides'][0]
        self.taskArnList = []
        for workerID in range(concCfg.numWorkers):
            # Construct the command args entry with the current workerID
            workerCmdArgs = ['-i', str(workerID), '--channaddr', channAddr]
            ctrOverrides['command'] = workerCmdArgs
            runTaskResponse = ecsClient.run_task(**runTaskParams)
            taskResp = runTaskResponse['tasks'][0]
            self.taskArnList.append(taskResp['taskArn'])

    def shutdown(self):
        """
        Shut down the workers and data channel
        """
        self.forceExit.set()
        self.checkTaskErrors()
        self.waitClusterTasksFinished()
        self.ecsClient.delete_cluster(cluster=self.clusterName)
        if hasattr(self, 'dataChan'):
            self.dataChan.shutdown()
        self.ecsClient.deregister_task_definition(taskDefinition=self.taskDefArn)

    def waitClusterTasksFinished(self):
        """
        Poll the given cluster until the number of tasks reaches zero
        """
        taskCount = self.getClusterTaskCount()
        startTime = time.time()
        timeout = 20
        timeExceeded = False
        while ((taskCount > 0) and (not timeExceeded)):
            time.sleep(5)
            taskCount = self.getClusterTaskCount()
            timeExceeded = (time.time() > (startTime + timeout))

        # If we exceeded timeout without reaching zero,
        # raise an exception
        if timeExceeded and (taskCount > 0):
            msg = ("Cluster task count timeout ({} seconds). ".format(timeout))
            raise PyShepSegTilingError(msg)

    def getClusterTaskCount(self):
        """
        Query the cluster, and return the number of tasks it has.
        This is the total of running and pending tasks.
        If the cluster does not exist, return None.
        """
        count = None
        clusterName = self.clusterName
        response = self.ecsClient.describe_clusters(clusters=[clusterName])
        if 'clusters' in response:
            for descr in response['clusters']:
                if descr['clusterName'] == clusterName:
                    count = (descr['runningTasksCount'] +
                             descr['pendingTasksCount'])
        return count

    def checkTaskErrors(self):
        response = self.ecsClient.describe_tasks(cluster=self.clusterName,
            tasks=self.taskArnList)
        for t in response['tasks']:
            if 'stoppedReason' in t:
                print('stoppedReason', t['stoppedReason'])


class SegSubprocMgr(SegmentationConcurrencyMgr):
    """
    Run tiled segmentation with concurrency based on subprocess workers.
    This is used only as a test bed for the NetworkDataChannel and external
    worker command, and should not be used in real life.
    """
    concurrencyType = CONC_SUBPROC

    def startWorkers(self):
        """
        Start all segmentation workers
        """
        self.processes = {}
        for workerID in range(self.concurrencyCfg.numWorkers):
            cmdWords = ["pyshepseg_segmentationworkercmd",
                "--idnum", str(workerID),
                "--channaddr", self.dataChan.addressStr()]
            self.processes[workerID] = subprocess.Popen(cmdWords,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True)


class NetworkDataChannel:
    """
    Single class to manage communication with workers running on different
    machines. Uses the facilities in multiprocessing.managers.

    Created from either the server or the client end, the constructor
    takes 
    """
    def __init__(self, inQue=None, segResultCache=None, forceExit=None,
            exceptionQue=None, segDataDict=None, readSemaphore=None,
            timings=None,
            hostname=None, portnum=None, authkey=None):
        class DataChannelMgr(multiprocessing.managers.BaseManager):
            pass

        if None not in (inQue, segResultCache):
            self.hostname = socket.gethostname()
            # Authkey is a big long random bytes string. Making one which is
            # also printable ascii.
            self.authkey = secrets.token_hex()

            self.inQue = inQue
            self.segResultCache = segResultCache
            self.forceExit = forceExit
            self.exceptionQue = exceptionQue
            self.readSemaphore = readSemaphore
            self.segDataDict = segDataDict
            self.timings = timings

            DataChannelMgr.register("get_inque", callable=lambda: self.inQue)
            DataChannelMgr.register("get_segresultcache",
                callable=lambda: self.segResultCache)
            DataChannelMgr.register("get_forceexit",
                callable=lambda: self.forceExit)
            DataChannelMgr.register("get_exceptionque",
                callable=lambda: self.exceptionQue)
            DataChannelMgr.register("get_segdatadict",
                callable=lambda: self.segDataDict)
            DataChannelMgr.register("get_readsemaphore",
                callable=lambda: self.readSemaphore)
            DataChannelMgr.register("get_timings",
                callable=lambda: self.timings)

            self.mgr = DataChannelMgr(address=(self.hostname, 0),
                                     authkey=bytes(self.authkey, 'utf-8'))

            self.server = self.mgr.get_server()
            self.portnum = self.server.address[1]
            self.threadPool = futures.ThreadPoolExecutor(max_workers=1)
            self.serverThread = self.threadPool.submit(
                self.server.serve_forever)
        elif None not in (hostname, portnum, authkey):
            DataChannelMgr.register("get_inque")
            DataChannelMgr.register("get_segresultcache")
            DataChannelMgr.register("get_forceexit")
            DataChannelMgr.register("get_exceptionque")
            DataChannelMgr.register("get_segdatadict")
            DataChannelMgr.register("get_readsemaphore")
            DataChannelMgr.register("get_timings")

            self.mgr = DataChannelMgr(address=(hostname, portnum),
                                     authkey=authkey)
            self.hostname = hostname
            self.portnum = portnum
            self.authkey = authkey
            self.mgr.connect()

            # Get the proxy objects.
            self.inQue = self.mgr.get_inque()
            self.segResultCache = self.mgr.get_segresultcache()
            self.forceExit = self.mgr.get_forceexit()
            self.exceptionQue = self.mgr.get_exceptionque()
            self.segDataDict = self.mgr.get_segdatadict()
            self.readSemaphore = self.mgr.get_readsemaphore()
            self.timings = self.mgr.get_timings()
        else:
            msg = ("Must supply either (inQue, segResultCache, etc.)" +
                   " or ALL of (hostname, portnum and authkey)")
            raise ValueError(msg)

    def shutdown(self):
        """
        Shut down the NetworkDataChannel in the right order. This should always
        be called explicitly by the creator, when it is no longer
        needed. If left to the garbage collector and/or the interpreter
        exit code, things are shut down in the wrong order, and the
        interpreter hangs on exit.

        I have tried __del__, also weakref.finalize and atexit.register,
        and none of them avoid these problems. So, just make sure you
        call shutdown explicitly, in the process which created the
        NetworkDataChannel.

        The client processes don't seem to care, presumably because they
        are not running the server thread. Calling shutdown on the client
        does nothing.

        """
        if hasattr(self, 'server'):
            self.server.stop_event.set()
            futures.wait([self.serverThread])
            self.threadPool.shutdown()

    def addressStr(self):
        """
        Return a single string encoding the network address of this channel
        """
        s = "{},{},{}".format(self.hostname, self.portnum, self.authkey)
        return s


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


class SegmentationResultCache:
    """
    Thread-safe cache for segmentation results, by tile. As each worker completes
    a tile, it adds it directly to this cache. The writing thread can then
    pop tiles out of this when required.
    """
    def __init__(self, colRowList, timeout=None):
        self.timeout = timeout
        self.lock = threading.Lock()
        self.cache = {}
        self.completionEvent = {}
        for (col, row) in colRowList:
            self.completionEvent[(col, row)] = threading.Event()

    def addResult(self, col, row, segResult):
        """
        Add a single segResult object to the cache, for the given (col, row)
        """
        with self.lock:
            key = (col, row)
            self.cache[key] = segResult
            self.completionEvent[key].set()

    def waitForTile(self, col, row):
        """
        Wait until the nominated tile is ready, and then pop it out of
        the cache.
        """
        key = (col, row)
        completed = self.completionEvent[key].wait(timeout=self.timeout)
        if completed:
            segResult = self.cache.pop(key)
            self.completionEvent[key].clear()
        else:
            segResult = None
        return segResult


class PyShepSegTilingError(Exception):
    pass


# Warning. Everything below this point is deprecated, and will likely be
# removed in a future version.
#

def calcHistogramTiled(segfile, maxSegId, writeToRat=True):
    """
    This function is now deprecated, and will probably be removed in
    a future version.

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

    Parameters
    ----------
      segfile : str or gdal.Dataset
        Segmentation image file. Can be either the file name string, or
        an open Dataset object.
      maxSegId : shepseg.SegIdType
        Maximum segment ID used
      writeToRat : bool
        If True, the completed histogram will be written to the image
        file's raster attribute table. If segfile was given as a Dataset
        object, it would therefore need to have been opened with update
        access.

    Returns
    -------
      hist : int ndarray (numSegments+1, )
        Histogram counts for each segment (index is segment ID number)

    """
    # Deprecation warning message.
    hInd = 13 * ' '     # Hanging indent in msg
    msg = '\n'.join([
        "The calcHistogramTiled function is obsolete, as histogram of ",
        hInd + "segmentation raster is now calculated as tiles are written. ",
        hInd + "It is deprecated, and will probably be removed in a future version"
    ])
    utils.deprecationWarning(msg)

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
    Fast function to increment counts for each segment ID in the given tile
    """
    (nrows, ncols) = tileData.shape
    for i in range(nrows):
        for j in range(ncols):
            segid = tileData[i, j]
            hist[segid] += 1
