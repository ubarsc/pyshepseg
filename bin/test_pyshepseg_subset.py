#!/usr/bin/env python

"""
Test harness for subsetting a segmented image
"""

#Copyright 2021 Neil Flood and Sam Gillingham. All rights reserved.
#
#Permission is hereby granted, free of charge, to any person 
#obtaining a copy of this software and associated documentation 
#files (the "Software"), to deal in the Software without restriction, 
#including without limitation the rights to use, copy, modify, 
#merge, publish, distribute, sublicense, and/or sell copies of the 
#Software, and to permit persons to whom the Software is furnished 
#to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be 
#included in all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
#EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
#OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
#IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR 
#ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF 
#CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION 
#WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from __future__ import print_function, division

import argparse
from pyshepseg import tiling

DFLT_OUTPUT_DRIVER = 'KEA'
GDAL_DRIVER_CREATION_OPTIONS = {'KEA' : [], 'HFA' : ['COMPRESS=YES']}

def getCmdargs():
    """     
    Get the command line arguments.
    """
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--infile", required=True,
        help="Input file")
    p.add_argument("-o", "--outfile", required=True,
        help="Output file")
    p.add_argument("--tlx", type=int, required=True,
        help="Top left x (in pixel coords) of the --infile to start extracting")
    p.add_argument("--tly", type=int,  required=True,
        help="Top left y (in pixel coords) of the --infile to start extracting")
    p.add_argument("--xsize", type=int,  required=True,
        help="X size of the window to extract")
    p.add_argument("--ysize", type=int,  required=True,
        help="Y size of the window to extract")
    p.add_argument("--origsegidcol", help="Name of column to write the original" +
                " segment ids")
    p.add_argument("-f", "--format", default=DFLT_OUTPUT_DRIVER, 
        choices=[DFLT_OUTPUT_DRIVER, "HFA"],
        help="Name of output GDAL format that supports RATs (default=%(default)s)")
    cmdargs = p.parse_args()
    return cmdargs

def main():
    cmdargs = getCmdargs()
    
    creationOptions = []
    if cmdargs.format in GDAL_DRIVER_CREATION_OPTIONS:
        creationOptions = GDAL_DRIVER_CREATION_OPTIONS[cmdargs.format]
    
    tiling.subsetImage(cmdargs.infile, cmdargs.outfile, cmdargs.tlx, cmdargs.tly, 
        cmdargs.xsize, cmdargs.ysize, cmdargs.format, creationOptions, 
        cmdargs.origsegidcol)
    
if __name__ == "__main__":
    main()
