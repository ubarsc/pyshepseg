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
import numpy
from osgeo import gdal

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
    # Notes. 
    # 1. This does not currently include a general mechanism
    #    for any image transformations. Is that a problem? 
    #    Perhaps a function could be passed? 
    # 2. Default subSampleProp = sqrt(ONEMILLION/(nRows*nCols))
    # 3. img = empty((nRows*subSampleProp, nCols*subSampleProp))
    #    band.ReadAsArray(buf_obj=img)
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
        self.tiles = []
        
    def addTile(self, xpos, ypos, xsize, ysize):
        self.tiles.append((xpos, ypos, xsize, ysize))
        
    def getNumTiles(self):
        return len(self.tiles)
        
    def getTile(self, n):
        return self.tiles[n]

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

            tileInfo.append(xpos, ypos, xsize, ysize)
            xpos += (tileSize - overlapSize)
            xtile += 1
            
        ypos += (tileSize - overlapSize)
        ytile += 1
        
    return tileInfo
    

def doTiledShepherdSegmentation(infile, outfile, tileSize, overlapSize=None,
        minSegmentSize=50, numClusters=60, bandNumbers=None, subsamplePcnt=None, 
        maxSpectralDiff='auto', imgNullVal=None, fixedKMeansInit=False,
        fourConnected=True, verbose=False):
    """
    Rough pseudocode sketch......
    """
    # km = fitSpectralClustersWholeFile() on whole infile
    #
    # Work out coords of tiles. There should be
    # an overlap wide enough so that no segment can extend from 
    # the middle to the edge of the overlap. (e.g. minSegmentSize/2)
    #
    # For each tile, do
    #     shepseg.doShepherdSegmentation(tile, kmeansObj=km)
    #
    # Stitch together output tiles into single mosaic, 
    # re-writing segment ID numbers to be unique. 
    
    kmeansObj, subSamplePcnt, imgNullVal = fitSpectralClustersWholeFile(infile, 
            numClusters, bandNumbers, subsamplePcnt, imgNullVal, fixedKMeansInit)
    
    inDs = gdal.Open(infile)
    
    if overlapSize is None:
        overlapSize = minSegmentSize / 2
        
    tileInfo = getTilesForFile(infile, tileSize, overlapSize)
    
    if bandNumbers is None:
        bandNumbers = range(1, inDs.RasterCount+1)
    
    for ntile in range(tileInfo.getNumTiles()):
        xpos, ypos, xsize, ysize = tileInfo.getTile(ntile)
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
        
        filename = 'tile_{}.kea'.format(ntile)
        outDrvr = gdal.GetDriverByName('KEA')
        
        if os.path.exists(filename):
            outDrvr.Delete(filename)

        outType = gdal.GDT_UInt32

        outDs = outDrvr.Create(filename, xsize, ysize, 1, outType)
        outDs.SetProjection(inDs.GetProjection())
        outDs.SetGeoTransform(inDs.GetGeoTransform())
        b = outDs.GetRasterBand(1)
        b.WriteArray(segResult.segimg)
        b.SetMetadataItem('LAYER_TYPE', 'thematic')
        b.SetNoDataValue(shepseg.SEGNULLVAL)

class PyShepSegTilingError(Exception): pass
