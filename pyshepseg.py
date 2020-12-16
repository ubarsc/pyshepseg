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

