#!/usr/bin/env python

import sys
import math
import argparse
from osgeo import gdal

DFLT_OUTPUT_DRIVER = 'KEA'
GDAL_DRIVER_CREATION_OPTIONS = {'KEA' : [], 'HFA' : ['COMPRESS=YES']}

def getCmdargs():
    """     
    Get the command line arguments.
    """
    p = argparse.ArgumentParser()
    p.add_argument('-s', '--size', type=int,
        help="Size in pixels that the tiles should be")
    p.add_argument('-o', '--overlap', type=int,
        help="Size in pixels of the overlap between tiles")
    p.add_argument("-f", "--format", default=DFLT_OUTPUT_DRIVER, 
        help="Name of output GDAL format (default=%(default)s)")
    p.add_argument("-b", '--base', 
        help=("base output filename to use. _X_Y.ext will be added to the end " +
            "with X is the across tile number and Y is the down tile number " +
            "and ext is the extension used by the given driver."))
    p.add_argument("-i", "--infile", help="Input mosaic")
            
    cmdargs = p.parse_args()
    
    if (cmdargs.size is None or cmdargs.overlap is None or cmdargs.base is None
            or cmdargs.infile is None):
        p.print_help()
        sys.exit()

    return cmdargs        

def main():
    cmdargs = getCmdargs()
    
    ds = gdal.Open(cmdargs.infile)

    outDriver = gdal.GetDriverByName(cmdargs.format)
    extension = outDriver.GetMetadataItem("DMD_EXTENSION")
    print(extension)

    creationOptions = None
    if cmdargs.format in GDAL_DRIVER_CREATION_OPTIONS:
        creationOptions = GDAL_DRIVER_CREATION_OPTIONS[cmdargs.format]

    yDone = False
    ypos = 0
    xtile = 0
    ytile = 0
    while not yDone:
    
        xDone = False
        xpos = 0
        xtile = 0
        ysize = cmdargs.size
        if (ypos + ysize) > ds.RasterYSize:
            ysize = ds.RasterYSize - ypos
            yDone = True
            if ysize == 0:
                break
    
        while not xDone:
            filename = cmdargs.base + '_{}_{}.{}'.format(xtile, ytile, extension)
            print(filename)
            
            xsize = cmdargs.size
            if (xpos + xsize) > ds.RasterXSize:
                xsize = ds.RasterXSize - xpos
                xDone = True
                if xsize == 0:
                    break
        
            options = gdal.TranslateOptions(format=cmdargs.format,
                            srcWin=[xpos, ypos, xsize, ysize],
                            creationOptions=creationOptions)
            print(xpos, ypos, xsize, ysize)
            
            gdal.Translate(filename, ds, options=options)
            
            xpos += (cmdargs.size - cmdargs.overlap)
            xtile += 1
            
        ypos += (cmdargs.size - cmdargs.overlap)
        ytile += 1
            
            

if __name__ == '__main__':
    main()
