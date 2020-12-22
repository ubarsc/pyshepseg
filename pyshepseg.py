"""
Python implementation of the image segmentation algorithm described
by Shepherd et al
    Shepherd, J., Bunting, P. and Dymond, J. (2019). 
        Operational Large-Scale Segmentation of Imagery Based on 
        Iterative Elimination. Remote Sensing 11(6).

Implemented using scikit-learn's K-Means algorithm, and using
numba compiled code for the other main components. 

"""
# Licence......

# Just in case anyone is trying to use this with Python-2
from __future__ import print_function, division

import time

import numpy
from sklearn.cluster import KMeans
from numba import njit

# This value used for null in both cluster ID and segment ID images
SEGNULLVAL = 0
MINSEGID = SEGNULLVAL + 1

def doShepherdSegmentation(img, numClusters=60, clusterSubsamplePcnt=1,
        minSegmentSize=50, maxSpectralDiff=0.1, imgNullVal=None,
        fourConnected=False, verbose=False):
    """
    Perform Shepherd segmentation in memory, on the given 
    multi-band img array.
    
    The img array has shape (nBands, nRows, nCols).
    numClusters and clusterSubsamplePcnt are passed
    through to makeSpectralClusters(). 
    minSegmentSize and maxSpectralDiff are passed through
    to eliminateSmallSegments(). 
    
    Default values are mostly as suggested by Shepherd et al. 
    
    If fourway is True, then use 4-way connectedness when clumping,
    otherwise use 8-way connectedness. ???? James and Pete seem to use
    4-way - why is this ????
    
    If imgNullVal is not None, then pixels with this value in 
    any band are set to zero (SEGNULLVAL) in the output segmentation. 
    
    Segment ID numbers start from 1. 
    
    """
    t0 = time.time()
    clusters = makeSpectralClusters(img, numClusters,
        clusterSubsamplePcnt, imgNullVal)
    if verbose:
        print("Kmeans, in", round(time.time()-t0, 1), "seconds")
    
    # Do clump
    t0 = time.time()
    (seg, maxSegId) = clump(clusters, SEGNULLVAL, fourConnected=fourConnected, 
        clumpId=MINSEGID)
    if verbose:
        print("Found", maxSegId, "clumps, in", round(time.time()-t0, 1), "seconds")
    
    # Make segment size array
    t0 = time.time()
    segSize = makeSegSize(seg)
    if verbose:
        print("Made seg sizes, in", round(time.time()-t0, 1), "seconds")
    
    # Eliminate small segments. Start with James' 
    # memory-efficient method for single pixel clumps. 
    t0 = time.time()
    (segSize, maxSegId, numElim) = eliminateSinglePixels(img, seg, segSize, 
        MINSEGID, maxSegId)
    if verbose:
        print("Eliminated", numElim, "single pixels, in", 
            round(time.time()-t0, 1), "seconds")
    
    
    if verbose:
        print("Final result has", maxSegId, "segments")
    
    # Return 
    #  (segment image array, segment spectral summary info, what else?)
    return seg


def makeSpectralClusters(img, numClusters, subsamplePcnt, imgNullVal):
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

    km = KMeans(n_clusters=numClusters)
    km.fit(xSample)

    # Predict on the whole image. In principle we could omit the nulls,
    # but it makes little difference to run time, and just adds complexity. 
    
    clustersFull = km.predict(xFull)
    del xFull, xNonNull, xSample
    clustersImg = clustersFull.reshape((nRows, nCols))
    
    # Make the cluster ID numbers start from 1, and use SEGNULLVAL
    # (i.e. zero) in null pixels
    clustersImg += 1
    if imgNullVal is not None:
        nullmask = (img == imgNullVal).any(axis=0)
        clustersImg[nullmask] = SEGNULLVAL

    return clustersImg


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
    
    ysize, xsize = img.shape
    output = numpy.zeros((ysize, xsize), dtype=numpy.uint32)
    search_list = numpy.empty((xsize * ysize, 2), dtype=numpy.uint32)
    
    searchIdx = 0
    
    # run through the image
    for y in range(ysize):
        for x in range(xsize):
            # check if we have visited this one before
            if img[y, x] != ignoreVal and output[y, x] == 0:
                val = img[y, x]
                searchIdx = 0
                search_list[searchIdx, 0] = y
                search_list[searchIdx, 1] = x
                searchIdx += 1
                output[y, x] = clumpId  # marked as visited
                
                while searchIdx > 0:
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
                                # add this one to the ones to search the neighbours
                                search_list[searchIdx, 0] = cy
                                search_list[searchIdx, 1] = cx
                                searchIdx += 1
                                
                clumpId += 1
                
    return (output, (clumpId-1))


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


def eliminateSinglePixels(img, seg, segSize, minSegId, maxSegId):
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
    numElim = _mergeSinglePixels(img, seg, segSize, segToElim)
    while numElim > 0:
        totalNumElim += numElim
        numElim = _mergeSinglePixels(img, seg, segSize, segToElim)
    
    # Now do a relabel.....
    segSize = _relabelSegments(seg, segSize, minSegId)
    maxSegId = len(segSize) + 1
    
    return (segSize, maxSegId, totalNumElim)


@njit
def _mergeSinglePixels(img, seg, segSize, segToElim):
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
                (ii, jj) = _findNearestNeighbourPixel(img, seg, i, j, segSize)
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
def _findNearestNeighbourPixel(img, seg, i, j, segSize):
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
def _relabelSegments(seg, segSize, minSegId):
    """
    The given seg array is an image of segment labels, with some 
    numbers unused, due to elimination of small segments. Go through 
    and find the unused numbers, and re-label segments above 
    these so that segment labels are contiguous. 
    
    Modifies the seg array in place.
    
    """
    oldNumSeg = len(segSize)
    subtract = numpy.zeros(oldNumSeg, dtype=numpy.uint32)
    
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
    
    newSegSize = segSize[segSize>0]
    return newSegSize


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
