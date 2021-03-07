"""
Python implementation of the image segmentation algorithm described
by Shepherd et al
    Shepherd, J., Bunting, P. and Dymond, J. (2019). 
        Operational Large-Scale Segmentation of Imagery Based on 
        Iterative Elimination. Remote Sensing 11(6).
    https://www.mdpi.com/2072-4292/11/6/658

Implemented using scikit-learn's K-Means algorithm, and using
numba compiled code for the other main components. 
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    https://numba.pydata.org/

Main entry point is the doShepherdSegmentation() function. 

Basic usage:
    from pyshepseg import shepseg
    
    # Read in a multi-band image as a single array, img,
    # of shape (nBands, nRows, nCols). 
    # Ensure that any null pixels are all set to a known 
    # null value in all bands. Failure to correctly identify 
    # null pixels can result in a poorer quality segmentation. 
    
    segRes = shepseg.doShepherdSegmentation(img, imgNullVal=nullVal)
    
The segimg attribute of the segRes object is an array
of segment ID numbers, of shape (nRows, nCols). 
    
Resulting segment ID numbers start from 1, and null pixels 
are set to zero. 

Efficient segment location:
    After segmentation, the location of individual segments can be
    found efficiently using the object returned by makeSegmentLocations(). 
    
    E.g. 
      segSize = shepseg.makeSegSize(segimg)
      segLoc = shepseg.makeSegmentLocations(segimg, segSize)
    
    This segLoc object is indexed by segment ID number (must be
    of type shepseg.SegIdType), and each element contains information
    about the pixels which are in that segment. This information 
    can be returned as a slicing object suitable to index the image array
    
    E.g. 
      segNdx = segLoc[segId].getSegmentIndices()
      vals = img[0][segNdx]
    
    This example would give an array of the pixel values from the first 
    band of the original image, for the given segment ID. 
    
    This can be a very efficient way to calculate per-segment 
    quantities. It can be used in pure Python code, or it can be used
    inside numba jit functions, for even greater efficiency. 

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

import time

import numpy
from sklearn.cluster import KMeans
from numba import njit
from numba.experimental import jitclass
from numba.core import types
from numba.typed import Dict

# A symbol for the data type used as a segment ID number
SegIdType = numpy.uint32

# This value used for null in both cluster ID and segment ID images
SEGNULLVAL = 0
MINSEGID = SEGNULLVAL + 1


class SegmentationResult(object):
    """
    Results of the segmentation process
    
    Attributes:
      segimg: Segment image, as numpy array (nRows, nCols)
        Elements are segment ID numbers (starting from 1)
      kmeans: The sklearn KMeans object, after fitting
      maxSpectralDiff: The value used to limit segment merging
      singlePixelsEliminated: Number of single pixels merged 
        to adjacent segments
      smallSegmentsEliminated: Number of small segments merged into 
        adjacent segments

    """
    segimg = None
    kmeans = None
    maxSpectralDiff = None
    singlePixelsEliminated = None
    smallSegmentsEliminated = None


def doShepherdSegmentation(img, numClusters=60, clusterSubsamplePcnt=1,
        minSegmentSize=50, maxSpectralDiff='auto', imgNullVal=None,
        fourConnected=True, verbose=False, fixedKMeansInit=False,
        kmeansObj=None, spectDistPcntile=50):
    """
    Perform Shepherd segmentation in memory, on the given 
    multi-band img array.
    
    The img array has shape (nBands, nRows, nCols).
    
    numClusters and clusterSubsamplePcnt are passed
    through to fitSpectralClusters(). 
    
    minSegmentSize is the minimum segment size (in pixels) which 
    will be left after eliminating small segments (except for
    segments which cannot be eliminated).
    
    maxSpectralDiff sets a limit on how different segments can be 
    and still be merged. It is given in the units of the spectral 
    space of img. If maxSpectralDiff is 'auto',
    a default value will be calculated from the spectral 
    distances between cluster centres, as a percentile of
    the distribution of these (spectDistPcntile). The value of
    spectDistPcntile should be lowered when segementing an image 
    with a larger range of spectral distances. 
    
    Default values are mostly as suggested by Shepherd et al. 
    
    If fourConnected is True, then use 4-way connectedness when clumping,
    otherwise use 8-way connectedness. 
    
    If imgNullVal is not None, then pixels with this value in 
    any band are set to zero (SEGNULLVAL) in the output 
    segmentation. If there are null values in the image array,
    it is important to give this null value, as it can strongly
    affect the initial spectral clustering, which in turn strongly
    affects the final segmentation. 
    
    If fixedKMeansInit is True, then choose a fixed set of 
    cluster centres to initialize the KMeans algorithm. This can
    be useful to provide strict determinacy of the results by
    avoiding sklearn's multiple random initial guesses. The default 
    is to allow sklearn to guess, which is good for avoiding 
    local minima. 
    
    By default, the spectral clustering step will be fitted using 
    the given img. However, if kmeansObj is not None, it is taken 
    to be a fitted instance of sklearn.cluster.KMeans, and will 
    be used instead. This is useful when enforcing a consistent 
    clustering across multiple tiles (see the pyshepseg.tiling 
    module for details). 
    
    Segment ID numbers start from 1. Zero is a NULL segment ID. 
    
    The return value is an instance of SegmentationResult class. 
    
    """
    t0 = time.time()
    if kmeansObj is not None:
        km = kmeansObj
    else:
        km = fitSpectralClusters(img, numClusters,
            clusterSubsamplePcnt, imgNullVal, fixedKMeansInit)
    clusters = applySpectralClusters(km, img, imgNullVal)
    if verbose:
        print("Kmeans, in", round(time.time()-t0, 1), "seconds")
    
    # Do clump
    t0 = time.time()
    (seg, maxSegId) = clump(clusters, SEGNULLVAL, fourConnected=fourConnected, 
        clumpId=MINSEGID)
    maxSegId = SegIdType(maxSegId - 1)
    if verbose:
        print("Found", maxSegId, "clumps, in", round(time.time()-t0, 1), "seconds")
    
    # Make segment size array
    segSize = makeSegSize(seg)
    
    # Eliminate small segments. Start with James' 
    # memory-efficient method for single pixel clumps. 
    t0 = time.time()
    oldMaxSegId = maxSegId
    eliminateSinglePixels(img, seg, segSize, MINSEGID, maxSegId, fourConnected)
    maxSegId = seg.max()
    numElimSinglepix = oldMaxSegId - maxSegId
    if verbose:
        print("Eliminated", numElimSinglepix, "single pixels, in", 
            round(time.time()-t0, 1), "seconds")

    maxSpectralDiff = autoMaxSpectralDiff(km, maxSpectralDiff, spectDistPcntile)

    t0 = time.time()
    numElimSmall = eliminateSmallSegments(seg, img, maxSegId, minSegmentSize, maxSpectralDiff,
        fourConnected, MINSEGID)
    if verbose:
        print("Eliminated", numElimSmall, "segments, in", round(time.time()-t0, 1), "seconds")
    
    if verbose:
        print("Final result has", seg.max(), "segments")
    
    segResult = SegmentationResult()
    segResult.segimg = seg
    segResult.kmeans = km
    segResult.maxSpectralDiff = maxSpectralDiff
    segResult.singlePixelsEliminated = numElimSinglepix
    segResult.smallSegmentsEliminated = numElimSmall
    return segResult


def fitSpectralClusters(img, numClusters, subsamplePcnt, imgNullVal,
    fixedKMeansInit):
    """
    First step of Shepherd segmentation. Use K-means clustering
    to create a set of "seed" segments, labelled only with
    their spectral cluster number. 
    
    The img array has shape (nBands, nRows, nCols).
    numClusters is the number of clusters for the KMeans
    algorithm to find (i.e. it is 'k'). 
    subsamplePcnt is the percentage of the pixels to actually use 
    for KMeans clustering. Shepherd et al find that only
    a very small percentage is required. 
    
    If imgNullVal is not None, then pixels in img with this value in 
    any band are set to segNullVal in the output. 
    
    if fixedKMeansInit is True, then use a simple algorithm to 
    determine the fixed set of initial cluster centres. Otherwise 
    allow the sklearn routine to choose its own initial guesses. 
    
    Return a fitted object of class sklearn.cluster.KMeans. This
    is suitable to use with the applySpectralClusters() function. 

    """
    (nBands, nRows, nCols) = img.shape

    # Re-organise the image data so it matches what sklearn
    # expects.
    xFull = numpy.transpose(img, axes=(1, 2, 0))
    xFull = xFull.reshape((nRows*nCols, nBands))

    if imgNullVal is not None:
        # Only use non-null values for fitting
        nonNull = (xFull != imgNullVal).all(axis=1)
        xNonNull = xFull[nonNull]
        del nonNull
    else:
        xNonNull = xFull
    skip = int(round(100./subsamplePcnt))
    xSample = xNonNull[::skip]
    del xFull, xNonNull

    # Note that we limit the number of trials that KMeans does, using 
    # the n_init argument. Multiple trials are used to avoid getting 
    # stuck in local minima, but 5 seems plenty, and this is the 
    # slowest part, so let's not get carried away. 
    numKmeansTrials = 5
    
    init = 'k-means++'      # This is sklearn's default
    if fixedKMeansInit:
        init = diagonalClusterCentres(xSample, numClusters)
        numKmeansTrials = 1
    km = KMeans(n_clusters=numClusters, n_init=numKmeansTrials, init=init)
    km.fit(xSample)
    
    return km


def applySpectralClusters(kmeansObj, img, imgNullVal):
    """
    Use the given KMeans object to predict spectral clusters on 
    a whole image array. 
    
    The kmeansObj is an instance of sklearn.cluster.KMeans, 
    as returned by fitSpectralClusters(). 
    
    The img array is a numpy array of the image to predict on,
    of shape (nBands, nRows, nCols). 
    
    Any pixels in img which have value imgNullVal will be set to
    SEGNULLVAL (i.e. zero) in the output cluster image.
    
    Return value is a numpy array of shape (nRows, nCols),
    with each element being the segment ID value for that pixel. 
    
    """

    # Predict on the whole image. In principle we could omit the nulls,
    # but it makes little difference to run time, and just adds complexity. 
    
    (nBands, nRows, nCols) = img.shape

    # Re-organise the image data so it matches what sklearn
    # expects.
    xFull = numpy.transpose(img, axes=(1, 2, 0))
    xFull = xFull.reshape((nRows*nCols, nBands))

    clustersFull = kmeansObj.predict(xFull)
    del xFull
    clustersImg = clustersFull.reshape((nRows, nCols))
    
    # Make the cluster ID numbers start from 1, and use SEGNULLVAL
    # (i.e. zero) in null pixels
    clustersImg += 1
    if imgNullVal is not None:
        nullmask = (img == imgNullVal).any(axis=0)
        clustersImg[nullmask] = SEGNULLVAL

    return clustersImg


def diagonalClusterCentres(xSample, numClusters):
    """
    Return an array of initial guesses at cluster centres. 
    This will be given to the KMeans constructor as the init
    parameter. 
    
    The given array xSample is the (numPoints, numBands) array
    ready to be used for fitting. 
    
    The centres are evenly spaced along the diagonal of 
    the bounding box of the data. The end points are placed 
    1 step in from the corners. 
    
    """
    (numPoints, numBands) = xSample.shape
    bandMin = xSample.min(axis=0)
    bandMax = xSample.max(axis=0)
    
    centres = numpy.empty((numClusters, numBands), dtype=xSample.dtype)
    
    step = (bandMax - bandMin) / (numClusters + 1)
    for i in range(numClusters):
        centres[i] = bandMin + (i+1) * step
    
    return centres


def autoMaxSpectralDiff(km, maxSpectralDiff, distPcntile):
    """
    Work out what to use as the maxSpectralDiff.

    If current value is 'auto', then return the median spectral
    distance between cluster centres from the KMeans clustering
    object km.

    If current value is None, return 10 times the largest distance
    between cluster centres (i.e. too large ever to make a difference)

    Otherwise, return the given current value.

    """
    # Calculate distances between pairs of cluster centres
    centres = km.cluster_centers_
    numClusters = centres.shape[0]
    numPairs = numClusters * (numClusters - 1) // 2
    clusterDist = numpy.full(numPairs, -1, dtype=numpy.float32)
    k = 0
    for i in range(numClusters-1):
        for j in range(i+1, numClusters):
            clusterDist[k] = numpy.sqrt(((centres[i] - centres[j])**2).sum())
            k += 1

    if maxSpectralDiff == 'auto':
        maxSpectralDiff = numpy.percentile(clusterDist, distPcntile)
    elif maxSpectralDiff is None:
        maxSpectralDiff = 10 * clusterDist.max()

    return maxSpectralDiff


@njit
def clump(img, ignoreVal, fourConnected=True, clumpId=1):
    """
    Implementation of clumping using Numba
    Uses the 4 connected algorithm if fourConnected is True,
    Otherwise 8 connected
    img should be an integer 2d array containing the data to be clumped.
    ignoreVal should be the no data value for the input
    clumpId is the start clump id to use    

    returns a 2d uint32 array containing the clump ids
    and the highest clumpid used + 1
    
    """
    
    # Prevent really large clumps, as they create a 
    # serious performance hit later. In initial testing without 
    # this limit, final segmentation had >99.9% of segments 
    # smaller than this, so this seems like a good size to stop. 
    MAX_CLUMP_SIZE = 10000
    
    ysize, xsize = img.shape
    output = numpy.zeros((ysize, xsize), dtype=SegIdType)
    search_list = numpy.empty((xsize * ysize, 2), dtype=numpy.uint32)
    
    searchIdx = 0
    
    # run through the image
    for y in range(ysize):
        for x in range(xsize):
            # check if we have visited this one before
            if img[y, x] != ignoreVal and output[y, x] == 0:
                val = img[y, x]
                clumpSize = 0
                searchIdx = 0
                search_list[searchIdx, 0] = y
                search_list[searchIdx, 1] = x
                searchIdx += 1
                output[y, x] = clumpId  # marked as visited
                
                while searchIdx > 0 and (clumpSize < MAX_CLUMP_SIZE):
                    # search the last one
                    searchIdx -= 1
                    sy = search_list[searchIdx, 0]
                    sx = search_list[searchIdx, 1]

                    # work out the 3x3 window to vist
                    tlx = sx - 1
                    if tlx < 0:
                        tlx = 0
                    tly = sy - 1
                    if tly < 0:
                        tly = 0
                    brx = sx + 1
                    if brx > xsize - 1:
                        brx = xsize - 1
                    bry = sy + 1
                    if bry > ysize - 1:
                        bry = ysize - 1

                    # do a '4 neighbour search'
                    for cx in range(tlx, brx+1):
                        for cy in range(tly, bry+1):
                            connected = not fourConnected or (cy == sy or cx == sx)
                            # don't have to check we are the middle
                            # cell since output will be != 0
                            # since we do that before we add it to search_list
                            if connected and (img[cy, cx] != ignoreVal and 
                                    output[cy, cx] == 0 and 
                                    img[cy, cx] == val):
                                output[cy, cx] = clumpId # mark as visited
                                clumpSize += 1
                                # add this one to the ones to search the neighbours
                                search_list[searchIdx, 0] = cy
                                search_list[searchIdx, 1] = cx
                                searchIdx += 1
                                
                clumpId += 1
                
    return (output, clumpId)


@njit
def makeSegSize(seg):
    """
    Return an array of segment sizes, from the given seg image. The
    returned array is indexed by segment ID. Each element is the 
    number of pixels in that segment. 
    """
    maxSegId = seg.max()
    segSize = numpy.zeros(maxSegId+1, dtype=numpy.uint32)
    (nRows, nCols) = seg.shape
    for i in range(nRows):
        for j in range(nCols):
            segSize[seg[i, j]] += 1

    return segSize


def eliminateSinglePixels(img, seg, segSize, minSegId, maxSegId, fourConnected):
    """
    Approximate elimination of single pixels, as suggested 
    by Shepherd et al (section 2.3, page 6). This step suggested as 
    an efficient way of removing a large number of segments which 
    are single pixels, by approximating the spectrally-nearest
    neighbouring segment with the spectrally-nearest neighouring
    pixel. 
    
    img is the original spectral image, of shape (nBands, nRows, nCols)
    seg is the image of segments, of shape (nRows, nCols)
    Segment ID numbers start at 1, and the largest is maxSegId. 
    
    Modifies seg array in place. 
    
    """
    # Array to store info on pixels to be eliminated.
    # Store (row, col, newSegId). 
    segToElim = numpy.zeros((3, maxSegId), dtype=seg.dtype)
    
    totalNumElim = 0
    numElim = mergeSinglePixels(img, seg, segSize, segToElim, fourConnected)
    while numElim > 0:
        totalNumElim += numElim
        numElim = mergeSinglePixels(img, seg, segSize, segToElim, fourConnected)
    
    # Now do a relabel.....
    relabelSegments(seg, segSize, minSegId)


@njit
def mergeSinglePixels(img, seg, segSize, segToElim, fourConnected):
    """
    Search for single-pixel segments, and decide which neighbouring
    segment they should be merged with. Finds all to eliminate,
    then performs merge on all selected. Modifies seg and
    segSize arrays in place, and returns the number of segments 
    eliminated. 
    
    """
    (nRows, nCols) = seg.shape
    numEliminated = 0

    for i in range(nRows):
        for j in range(nCols):
            segid = seg[i, j]
            if segSize[segid] == 1:
                (ii, jj) = findNearestNeighbourPixel(img, seg, i, j, 
                        segSize, fourConnected)
                # Record the new segment ID for the current pixel
                if (ii >= 0 and jj >= 0):
                    segToElim[0, numEliminated] = i
                    segToElim[1, numEliminated] = j
                    segToElim[2, numEliminated] = seg[ii, jj]
                    numEliminated += 1
    
    # Now do eliminations, updating the seg array and the 
    # segSize array in place. 
    for k in range(numEliminated):
        r = segToElim[0, k]
        c = segToElim[1, k]
        newSeg = segToElim[2, k]
        oldSeg = seg[r, c]
        
        seg[r, c] = newSeg
        segSize[oldSeg] = 0
        segSize[newSeg] += 1

    return numEliminated


@njit
def findNearestNeighbourPixel(img, seg, i, j, segSize, fourConnected):
    """
    For the (i, j) pixel, choose which of the neighbouring
    pixels is the most similar, spectrally. 
    
    Returns tuple (ii, jj) of the row and column of the most
    spectrally similar neighbour, which is also in a 
    clump of size > 1. If none is found, return (-1, -1)
    """
    (nBands, nRows, nCols) = img.shape
    
    minDsqr = -1
    ii = jj = -1
    # Cope with image edges
    (iiiStrt, iiiEnd) = (max(i-1, 0), min(i+1, nRows-1))
    (jjjStrt, jjjEnd) = (max(j-1, 0), min(j+1, nCols-1))
    
    for iii in range(iiiStrt, iiiEnd+1):
        for jjj in range(jjjStrt, jjjEnd+1):
            connected = ((not fourConnected) or ((iii == i) or (jjj == j)))
            if connected:
                segNbr = seg[iii, jjj]
                if segSize[segNbr] > 1:
                    # Euclidean distance in spectral space. Note that because 
                    # we are only interested in the order, we don't actually 
                    # need to do the sqrt (which is expensive)
                    dSqr = ((img[:, i, j] - img[:, iii, jjj]) ** 2).sum()
                    if minDsqr < 0 or dSqr < minDsqr:
                        minDsqr = dSqr
                        ii = iii
                        jj = jjj
    
    return (ii, jj)


@njit
def relabelSegments(seg, segSize, minSegId):
    """
    The given seg array is an image of segment labels, with some 
    numbers unused, due to elimination of small segments. Go through 
    and find the unused numbers, and re-label segments above 
    these so that segment labels are contiguous. 
    
    Modifies the seg array in place. The segSize array is not 
    updated, and should be recomputed. 
    
    """
    oldNumSeg = len(segSize)
    subtract = numpy.zeros(oldNumSeg, dtype=SegIdType)
    
    # For each segid with a count of zero (i.e. it is unused), we 
    # increase the amount by which segid numbers above this should 
    # be decremented
    for k in range(minSegId+1, oldNumSeg):
        subtract[k] = subtract[k-1]
        if segSize[k-1] == 0:
            subtract[k] += 1
    
    # Now decrement the segid of every pixel
    (nRows, nCols) = seg.shape
    for i in range(nRows):
        for j in range(nCols):
            oldSegId = seg[i, j]
            newSegId = oldSegId - subtract[oldSegId]
            seg[i, j] = newSegId
    

@njit
def buildSegmentSpectra(seg, img, maxSegId):
    """
    Build an array of the spectral statistics for each segment. 
    Return an array of shape
        (numSegments+1, numBands)
    where each row is the entry for that segment ID, and each 
    column is the sum of the spectral values for that band. 
    The zero-th entry is empty, as zero is not a valid
    segment ID. 
    
    """
    (nBands, nRows, nCols) = img.shape
    spectSum = numpy.zeros((maxSegId+1, nBands), dtype=numpy.float32)

    for i in range(nRows):
        for j in range(nCols):
            segid = seg[i, j]
            for k in range(nBands):
                spectSum[segid, k] += img[k, i, j]

    return spectSum


# This data structure is used to store the locations of
# every pixel, indexed by the segment ID. This means we can
# quickly find all the pixels belonging to a particular segment.
spec = [('idx', types.uint32), ('rowcols', types.uint32[:,:])]
@jitclass(spec)
class RowColArray(object):
    def __init__(self, length):
        self.idx = 0
        self.rowcols = numpy.empty((length, 2), dtype=numpy.uint32)

    def append(self, row, col):
        self.rowcols[self.idx, 0] = row
        self.rowcols[self.idx, 1] = col
        self.idx += 1
    
    def getSegmentIndices(self):
        """
        Return the row and column numbers of the segment pixels
        as a tuple, suitable for indexing the image array. 
        This supports selection of all pixels for a given segment. 
        """
        return (self.rowcols[:, 0], self.rowcols[:, 1])

RowColArray_Type = RowColArray.class_type.instance_type

@njit
def makeSegmentLocations(seg, segSize):
    """
    Create a data structure to hold the locations of all pixels
    in all segments.
    """
    d = Dict.empty(key_type=types.uint32, value_type=RowColArray_Type)
    numSeg = len(segSize)
    for segid in range(MINSEGID, numSeg):
        numPix = segSize[segid]
        obj = RowColArray(numPix)
        d[SegIdType(segid)] = obj

    (nRows, nCols) = seg.shape
    for row in range(nRows):
        for col in range(nCols):
            segid = seg[row, col]
            if segid != SEGNULLVAL:
                d[segid].append(row, col)

    return d


@njit
def eliminateSmallSegments(seg, img, maxSegId, minSegSize, maxSpectralDiff, 
        fourConnected, minSegId):
    """
    Eliminate small segments. Start with smallest, and merge
    them into spectrally most similar neighbour. Repeat for 
    larger segments. 
    
    minSegSize is the smallest segment which will NOT be eliminated
    
    """
    spectSum = buildSegmentSpectra(seg, img, maxSegId)
    segSize = makeSegSize(seg)
    segLoc = makeSegmentLocations(seg, segSize)

    # A list of the segment ID numbers to merge with. The i-th
    # element is the segment ID to merge segment 'i' into
    mergeSeg = numpy.empty(maxSegId+1, dtype=SegIdType)
    mergeSeg.fill(SEGNULLVAL)

    # Range of seg id numbers, as SegIdType, suitable as indexes into segloc
    segIdRange = numpy.arange(minSegId, maxSegId+1, dtype=SegIdType)

    # Start with smallest segments, move through to just 
    # smaller than minSegSize (i.e. minSegSize is smallest 
    # which will NOT be eliminated)
    numElim = 0
    for targetSize in range(1, minSegSize):
        countTargetSize = numpy.count_nonzero(segSize == targetSize)
        prevCount = -1
        # Use multiple passes to eliminate segments of this size. A 
        # single pass can leave segments unmerged, due to the rule about
        # only merging with neighbours larger than current. 
        # Note the use of MAXPASSES, just in case, as we hate infinite loops. 
        # A very small number can still be left unmerged, if surrounded by 
        # null segments. 
        (numPasses, MAXPASSES) = (0, 10)
        while (countTargetSize != prevCount) and (numPasses < MAXPASSES):
            prevCount = countTargetSize

            for segId in segIdRange:
                if segSize[segId] == targetSize:
                    mergeSeg[segId] = findMergeSegment(segId, segLoc, 
                        seg, segSize, spectSum, maxSpectralDiff, fourConnected)

            # Carry out the merges found above
            for segId in segIdRange:
                if mergeSeg[segId] != SEGNULLVAL:
                    doMerge(segId, mergeSeg[segId], seg, segSize, segLoc,
                        spectSum)
                    mergeSeg[segId] = SEGNULLVAL
                    numElim += 1

            countTargetSize = numpy.count_nonzero(segSize == targetSize)
            numPasses += 1

    relabelSegments(seg, segSize, minSegId)
    return numElim


@njit
def findMergeSegment(segId, segLoc, seg, segSize, spectSum, maxSpectralDiff,
                fourConnected):
    """
    For the given segId, find which neighboring segment it 
    should be merged with. The chosen merge segment is the one
    which is spectrally most similar to the given one, as
    measured by minimum Euclidean distance in spectral space. 
    """
    bestNbrSeg = SEGNULLVAL
    bestDistSqr = 0.0    # This value is never used

    (nRows, nCols) = seg.shape
    segRowcols = segLoc[segId].rowcols
    numPix = len(segRowcols)
    # Mean spectral bands
    spect = spectSum[segId] / numPix
    
    for k in range(numPix):
        (i, j) = segRowcols[k]
        for ii in range(max(i-1, 0), min(i+2, nRows)):
            for jj in range(max(j-1, 0), min(j+2, nCols)):
                connected = (not fourConnected) or (ii == i or jj == j)
                nbrSegId = seg[ii, jj]
                if (connected and (nbrSegId != segId) and 
                        (nbrSegId != SEGNULLVAL) and
                        (segSize[nbrSegId] > segSize[segId])):
                    nbrSpect = spectSum[nbrSegId] / segSize[nbrSegId]
                    
                    distSqr = ((spect - nbrSpect) ** 2).sum()
                    if ((bestNbrSeg == SEGNULLVAL) or (distSqr < bestDistSqr)):
                        bestDistSqr = distSqr
                        bestNbrSeg = nbrSegId

    if bestDistSqr > maxSpectralDiff**2:
        bestNbrSeg = SEGNULLVAL

    return bestNbrSeg


@njit
def doMerge(segId, nbrSegId, seg, segSize, segLoc, spectSum):
    """
    Carry out a single merge. The segId segment is merged to the
    neighbouring nbrSegId. 
    Modifies seg, segSize, segLoc and spectSum in place. 
    """
    segRowcols = segLoc[segId].rowcols
    numPix = len(segRowcols)
    nbrSegRowcols = segLoc[nbrSegId].rowcols
    nbrNumPix = len(nbrSegRowcols)
    mergedNumPix = numPix + nbrNumPix
    
    # New segLoc entry for merged segment
    mergedSegLoc = RowColArray(mergedNumPix)
    # Copy over the existing rowcols
    for k in range(nbrNumPix):
        (r, c) = nbrSegRowcols[k]
        mergedSegLoc.append(r, c)
    
    # Append the segment being merged
    for k in range(numPix):
        (r, c) = segRowcols[k]
        seg[r, c] = nbrSegId
        mergedSegLoc.append(r, c)

    # Replace the previous segLoc entry, and delete the one we merged
    segLoc[nbrSegId] = mergedSegLoc
    segLoc.pop(segId)
    
    # Update the spectral sums for the two segments
    numBands = spectSum.shape[1]
    for m in range(numBands):
        spectSum[nbrSegId, m] += spectSum[segId, m]
        spectSum[segId, m] = 0
    # Update the segment sizes
    segSize[nbrSegId] += segSize[segId]
    segSize[segId] = 0
