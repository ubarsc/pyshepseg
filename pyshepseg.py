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
from sklearn import KMeans
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



def makeSpectralClusters(img, numClusters, subsamplePcnt):
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

