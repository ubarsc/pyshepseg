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


def doShepherdSegmentation(img, numClusters=60, clusterSubsamplePcnt=1):
    """
    Perform Shepherd segmentation in memory, on the given 
    multi-band img array.
    
    The img array has shape (nBands, nRows, nCols).
    
    """
    clusters = makeSpectralClusters(img, 
        numClusters=numClusters,
        subsamplePcnt=clusterSubsamplePcnt)
    



def makeSpectralClusters(img, numClusters=60, subsamplePcnt=1):
    """
    First step of Shepherd segmentation. Use K-means clustering
    to create a set of "seed" segments, labelled only with
    their spectral cluster number. 
    
    The img array has shape (nBands, nRows, nCols).

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

