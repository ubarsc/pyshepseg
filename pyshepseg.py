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

import numpy
from sklearn.cluster import KMeans
from numba import jit


def doShepherdSegmentation(img, numClusters=60, clusterSubsamplePcnt=1,
        minSegmentSize=10, maxSpectralDiff=0.1):
    """
    Perform Shepherd segmentation in memory, on the given 
    multi-band img array.
    
    The img array has shape (nBands, nRows, nCols).
    numClusters and clusterSubsamplePcnt are passed
    through to makeSpectralClusters(). 
    minSegmentSize and maxSpectralDiff are passed through
    to eliminateSmallSegments(). 
    
    Default values are mostly as suggested by Shepherd et al. 
    
    """
    clusters = makeSpectralClusters(img, numClusters,
        clusterSubsamplePcnt)
    
    # Do clump
    
    # Eliminate small segments. If we wish, start with James' 
    # memory-efficient method for single pixel clumps. 
    
    # Re-label segments to contiguous numbering. 
    
    # Return 
    #  (segment image array, segment spectral summary info, what else?)



def makeSpectralClusters(img, numClusters, subsamplePcnt, fourway=False):
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
    
    If fourway is True, then use 4-way connectedness when clumping.
    Default is to use 8-way connectedness. ???? James and Pete seem to use
    4-way - why is this ????

    """
    (nBands, nRows, nCols) = img.shape

    # Re-organise the image data so it matches what sklearn
    # expects.
    xFull = numpy.transpose(img, axes=(1, 2, 0))
    xFull = xFull.reshape((nRows*nCols, nBands))

    skip = int(round(100./subsamplePcnt))
    xSample = xFull[::skip]

    km = KMeans(n_clusters=numClusters)
    km.fit(xSample)

    clustersFull = km.predict(xFull)
    del xFull, xSample
    clustersImg = clustersFull.reshape((nRows, nCols))

    return clustersImg



# Sam's numba-based clump routine to go here


def eliminateSinglePixels(img, seg, maxSegId):
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
    # Make an array of segment sizes (in pixels), indexed by segment ID
    (segSize, _) = numpy.histogram(seg, bins=range(maxSegId+2))
    # Save some space
    if segSize.max() < numpy.uint32(-1):
        segSize = segSize.astype(numpy.uint32)

    # Array to store info on pixels to be eliminated.
    # Store (row, col, newSegId). 
    segToElim = numpy.zeros((3, maxSegId), dtype=seg.dtype)
    
    numElim = _mergeSinglePixels(img, seg, segSize, segToElim)
    while numElim > 0:
        numElim = _mergeSinglePixels(img, seg, segSize, segToElim)
    
    # Now do a relabel.....

    

@jit(nopython=True)
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



@jit(nopython=True)
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
