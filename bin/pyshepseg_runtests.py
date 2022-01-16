#!/usr/bin/env python
"""
Run tests of the pyshepseg package. Use a generated dataset, sufficient 
to allow a meaningful segmentation. Note that the generated dataset is not 
sufficiently complex to be a strong test of the Shepherd segmentation
algorithm, but merely whether this implementation of the algorithm is
coded to behave sensibly. 

"""
import os
import argparse

import numpy

from osgeo import gdal

from pyshepseg import shepseg, tiling, utils

# This is a list of (x, y) coordinate pairs, representing centres of
# some test segments. These were initially generated randomly, but
# are saved here so we can use them reproduceably for our testing. 
# Note that (x, y) will also be used as (column, row) in the 
# image array, because there is no point in creating a separate
# world coordinate system (it would make no difference to the things
# being tested). 
initialCentres = [(116, 3495), (142, 3100), (236, 6033), (290, 796), (297, 6152), (310, 5318), (409, 5867), (410, 2125), (442, 2913), 
    (472, 1135), (486, 5296), (628, 667), (655, 2677), (672, 4001), (677, 5513), (736, 3720), (913, 3552), (1056, 347), (1085, 3391), 
    (1121, 6623), (1150, 1906), (1196, 5663), (1694, 3244), (1761, 2172), (1761, 7460), (1882, 6151), (1893, 626), (2014, 433), (2065, 3157), 
    (2132, 378), (2161, 2352), (2200, 7485), (2393, 5191), (2489, 2519), (2508, 1575), (2509, 7089), (2599, 3151), (2645, 2672), (2782, 3380), 
    (2906, 3676), (3072, 2934), (3133, 3418), (3188, 1653), (3624, 7812), (3661, 3603), (3694, 2929), (3759, 3418), (4155, 630), (4233, 4753), 
    (4423, 1377), (4427, 6635), (4462, 7392), (4715, 6908), (4856, 2559), (4898, 3371), (5051, 2268), (5064, 5969), (5071, 2019), (5107, 3533), 
    (5172, 5478), (5294, 4210), (5305, 1512), (5310, 2846), (5365, 3715), (5447, 6215), (5513, 5017), (5549, 297), (5579, 4076), (5623, 5044), 
    (5688, 3614), (5728, 1802), (5747, 7801), (5758, 4377), (5779, 4148), (5784, 3239), (5812, 5091), (5862, 4664), (5897, 4963), (6299, 4702), 
    (6320, 6936), (6462, 2844), (6615, 4979), (6726, 5970), (6754, 7652), (6765, 714), (6826, 3162), (6827, 3770), (6844, 1170), (6884, 226), 
    (7023, 213), (7094, 6472), (7157, 647), (7196, 7710), (7293, 7588), (7495, 5912), (7693, 3966), (7718, 7759), (7737, 6002), (7745, 1347), 
    (7889, 2850)]

# Shape of the image we will be working with. 
(nRows, nCols) = (8000, 8000)
# Number of bands in constructed multispectral image
NBANDS = 3


def getCmdargs():
    """
    Get command line arguments
    """
    p = argparse.ArgumentParser(description="""
        Run tests of the software, using internally generated data.
        This is mainly intended for checking whether code changes
        have broken anything. It is not a rigorous test of the 
        Shepherd algorithm. 
    """)
    p.add_argument("--keep", default=False, action="store_true",
        help="Keep test data files (default will delete)")
    p.add_argument("--knownseg", help=("Use this file as the initial "+
        "known segmentation, instead of generating one. Helpful "+
        " with --keep, to save time in repeated tests"))
    return p.parse_args()


def main():
    """
    Main routine
    """
    cmdargs = getCmdargs()
    
    truesegfile = "tmp_trueseg.kea"
    if cmdargs.knownseg is not None:
        truesegfile = cmdargs.knownseg
    imagefile = "tmp_image.kea"
    outsegfile = "tmp_seg.kea"
    tmpdatafiles = [imagefile, outsegfile]
    if cmdargs.knownseg is None:
        tmpdatafiles.append(truesegfile)

    if cmdargs.knownseg is None:
        print("Generating known segmentation {}".format(truesegfile))
        generateTrueSegments(truesegfile)
    print("Creating known multi-spectral image {}".format(imagefile))
    createMultispectral(truesegfile, imagefile)
    
    # Ensure enough clusters that we have a different cluster for each
    # segment in the generated image. We have not guarded against neighbours
    # being similar, so this will prevent neighbouring segments being 
    # merged too easily. 
    numClusters = len(initialCentres)

    print("Doing tiled segmentation, creating {}".format(outsegfile))
    # Note that we use fourConnected=False, to avoid disconnecting the 
    # pointy ends of long thin slivers, which can arise due to how we
    # generated the original segments. 
    segResults = tiling.doTiledShepherdSegmentation(imagefile, outsegfile, 
        numClusters=numClusters, fixedKMeansInit=True, fourConnected=False)
    
    (meanColNames, stdColNames) = makeRATcolumns(segResults, outsegfile, imagefile)

    pcntMatch = checkSegmentation(imagefile, outsegfile, meanColNames,
        stdColNames)
    
    print("Perfect match on {}% of pixels".format(pcntMatch))
    
    print("Adding colour table to {}".format(outsegfile))
    utils.writeColorTableFromRatColumns(outsegfile, meanColNames[0], 
        meanColNames[1], meanColNames[2])

    if not cmdargs.keep:
        print("Removing generated data")
        for fn in tmpdatafiles:
            drvr = gdal.IdentifyDriver(fn)
            drvr.Delete(fn)


# The basis of the test data will be a set of "true" segments. From 
# these we will generate multi-spectral data which represents these
# segments, and the tests will then use the pyshepseg package to 
# re-create the original segments from the multi-spectral data. 

def generateTrueSegments(outfile):
    """
    This routine generates the true segments from the initial segment
    centres hardwired in the initialCentres variable. 
    
    Each pixel is in the segment for its closest centre coordinate. 
    
    Saves the generated segment layer into the given raster filename,
    with format KEA. 
    
    """
    segArray = numpy.zeros((nRows, nCols), dtype=shepseg.SegIdType)
    segArray.fill(shepseg.SEGNULLVAL)
    
    minDist = numpy.zeros((nRows, nCols), dtype=numpy.float32)
    # Initial distance much bigger than whole array, so actual centres 
    # will all be closer
    minDist.fill(10*nCols)
    
    # For each pixel, its (x, y) position, to use in calculating distances
    (xGrid, yGrid) = numpy.mgrid[:nRows, :nCols]

    numCentres = len(initialCentres)
    for i in range(numCentres):
        (x, y) = initialCentres[i]
        dist = numpy.sqrt((xGrid - x)**2 + (yGrid - y)**2)
        minNdx = (dist < minDist)
        
        segId = i + 1
        segArray[minNdx] = segId
        # For each pixel, update the distance to the closest centre
        minDist[minNdx] = dist[minNdx]
    
    # Put in a margin of nulls all round, so that we can also test 
    # that null handling is working properly
    m = 10
    segArray[:m, :] = shepseg.SEGNULLVAL
    segArray[-m:, :] = shepseg.SEGNULLVAL
    segArray[:, :m] = shepseg.SEGNULLVAL
    segArray[:, -m:] = shepseg.SEGNULLVAL
    
    # Save to a KEA file
    drvr = gdal.GetDriverByName('KEA')
    if os.path.exists(outfile):
        drvr.Delete(outfile)
    ds = drvr.Create(outfile, nCols, nRows, bands=1, eType=gdal.GDT_UInt32)
    ds.SetGeoTransform((0, 1, 0, 0, 0, -1))
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(shepseg.SEGNULLVAL)
    band.WriteArray(segArray)
    del ds


def createPallete(numSeg):
    """
    Return a "pallete" of 3-band colours, one for each segment. 
    
    The colours are just made up, and have no particular meaning. The
    main criterion is that they be distinct, sufficiently so that 
    two adjacent segments should always come out in different colours. 
    
    Return value is an array of shape (numSeg, 3). Individual colour 
    values are in the range [0, 10000], and so the array has type uint16. 
    
    Note that the index into this array would be (segmentID - 1). 
    """
    MINVAL = 0
    MAXVAL = 10000
    step = (MAXVAL - MINVAL) / (numSeg - 1)
    mid = numSeg / 2
    
    c = numpy.zeros((numSeg, NBANDS), dtype=numpy.uint16)
    
    for i in range(numSeg):
        c[i, 0] = round(MINVAL + i * step)
        c[i, 1] = round(MAXVAL - i * step)
        if i < mid:
            c[i, 2] = round(MINVAL + i * 2 * step)
        else:
            c[i, 2] = round(MAXVAL - (i-mid) * 2 * step)
    
    return c


def createMultispectral(truesegfile, outfile):
    """
    Reads the given true segment file, and generates a multi-spectral
    image which should segment in a similar way. 
    """
    trueseg = readSeg(truesegfile)
    numSeg = trueseg.max()
    outNull = 2**16 - 1
    
    pallete = createPallete(numSeg)

    (nRows, nCols) = trueseg.shape
    outBand = numpy.zeros(trueseg.shape, dtype=numpy.uint16)
    nullNdx = (trueseg == shepseg.SEGNULLVAL)

    segSize = shepseg.makeSegSize(trueseg)
    segLoc = shepseg.makeSegmentLocations(trueseg, segSize)

    # Open output file    
    drvr = gdal.GetDriverByName('KEA')
    if os.path.exists(outfile):
        drvr.Delete(outfile)
    ds = drvr.Create(outfile, nCols, nRows, bands=NBANDS, eType=gdal.GDT_UInt16)

    # Generate each output band, writing as we go. 
    for i in range(NBANDS):
        for segId in segLoc:
            segNdx = segLoc[shepseg.SegIdType(segId)].getSegmentIndices()
            outBand[segNdx] = pallete[segId-1][i]
        outBand[nullNdx] = outNull

        b = ds.GetRasterBand(i+1)
        b.SetNoDataValue(outNull)
        b.WriteArray(outBand)
        b.FlushCache()
    ds.FlushCache()
    del ds


def readSeg(segfile):
    """
    Open and read the given segfile. Return an image array of the 
    segment ID values
    """
    ds = gdal.Open(segfile)
    band = ds.GetRasterBand(1)
    seg = band.ReadAsArray().astype(shepseg.SegIdType)
    return seg


def makeRATcolumns(segResults, outsegfile, imagefile):
    """
    Add some columns to the RAT, with useful per-segment statistics
    """
    # Calculate Histogram column for segfile
    tiling.calcHistogramTiled(outsegfile, segResults.maxSegId, writeToRat=True)

    # Calculate per-segment mean and stddev for all bands, and store in the RAT
    meanColNames = []
    stdColNames = []
    for i in range(NBANDS):
        meanCol = "Band_{}_mean".format(i+1)
        stdCol = "Band_{}_stddev".format(i+1)
        meanColNames.append(meanCol)
        stdColNames.append(stdCol)
        statsSelection = [(meanCol, "mean"), (stdCol, "stddev")]
        tiling.calcPerSegmentStatsTiled(imagefile, i+1, outsegfile, 
            statsSelection)
    
    return (meanColNames, stdColNames)


def checkSegmentation(imgfile, segfile, meanColNames, stdColNames):
    """
    Check whether the given segmentation of the given image file 
    is "correct", by some measure(s). 

    """
    seg = readSeg(segfile)
    nonNull = (seg != shepseg.SEGNULLVAL)

    # The tolerance to use for testing equality. The spectral differences
    # between segments are much larger than 1, but we are comparing
    # single pixel spectra with the segment means. If a single pixel
    # is incorrectly placed, then the segment mean will be only slightly 
    # affected, but the pixel spectra will disagree to a much greater
    # amount. So, a tolerance of almost 1 will still detect the single
    # pixels which are incorrectly placed. 
    TOL = 0.5

    colourMatch = None
    ds = gdal.Open(imgfile)
    for i in range(NBANDS):
        bandobj = ds.GetRasterBand(i+1)
        img = bandobj.ReadAsArray()

        segmeans = readColumn(segfile, meanColNames[i])
        segstddev = readColumn(segfile, stdColNames[i])

        # An img of the segmean for this band, for each pixel. 
        segColour = segmeans[seg]

        diff = numpy.absolute(img - segColour)
        diff[~nonNull] = 0    # Do this properly!!!!!
        ndx = numpy.where(diff>TOL)
        # Per-pixel, True when segment mean matches image colour for this band
        colourMatch_band = (diff < TOL)

        # Accumulate matches across bands. Ultimately, it is
        # a match if all bands match. 
        if colourMatch is None:
            colourMatch = colourMatch_band
        else:
            colourMatch = (colourMatch & colourMatch_band)

    numColourMatch = numpy.count_nonzero(colourMatch)

    # Rough check that nulls are in the right places
    imgnullval = bandobj.GetNoDataValue()
    nullMatch = (img[~nonNull] == imgnullval)
    numNullMatch = numpy.count_nonzero(nullMatch)

    # Percentage of pixels which match, either full colour match, or null
    numPix = len(colourMatch.flatten()) + len(nullMatch)
    pcntMatch = 100 * (numColourMatch + numNullMatch) / numPix

    return pcntMatch


def readColumn(segfile, colName):
    """
    Read the given column from the given segmentation image file.
    Return an array of the column values. 
    """
    ds = gdal.Open(segfile)
    band = ds.GetRasterBand(1)
    attrTbl = band.GetDefaultRAT()
    numCols = attrTbl.GetColumnCount()
    colNameList = [attrTbl.GetNameOfCol(i) for i in range(numCols)]
    colNdx = colNameList.index(colName)
    col = attrTbl.ReadAsArray(colNdx)
    
    return col


if __name__ == "__main__":
    main()
