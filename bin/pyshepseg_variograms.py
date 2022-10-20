#!/usr/bin/env python

"""
Test harness for tilingstats.calcPerSegmentSpatialStatsTiled().
Calculates the given number of variograms and saves them to the
segmented file's RAT.
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

import argparse

from osgeo import gdal

from pyshepseg import tilingstats


def getCmdargs():
    """     
    Get the command line arguments.
    """
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--infile", required=True,
        help="Input file to collect stats from")
    p.add_argument("-s", "--segfile", required=True,
        help="File from segmentation. Note: stats is written into the RAT in this file")
    p.add_argument("-n", "--numvariograms", required=True,
        choices=[x for x in range(1, 10)], type=int,
        help="Number of variograms to calculate")
    cmdargs = p.parse_args()
    return cmdargs

                    
def main():
    cmdargs = getCmdargs()
    cols = []
    for n in range(cmdargs.numvariograms):
        cols.append(("variogram{}".format(n + 1), gdal.GFT_Real))
        
    # find the appropriate user function
    funcName = 'userFuncVariogram{}'.format(cmdargs.numvariograms)
    userFunc = getattr(tilingstats, funcName)

    tilingstats.calcPerSegmentSpatialStatsTiled(cmdargs.infile, 1, 
        cmdargs.segfile, cols, userFunc)


if __name__ == '__main__':
    main()
