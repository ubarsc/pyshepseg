#!/usr/bin/env python3

"""
Script that stiches all the tiles together by calling
tiling.doTiledShepherdSegmentation_finalize().

Uplaods the resulting segmentation to S3.
"""

import io
import os
import json
import pickle
import resource
import argparse
import tempfile
import shutil
import boto3
from pyshepseg import tiling, tilingstats, utils
from osgeo import gdal

gdal.UseExceptions()


def getCmdargs():
    """
    Process the command line arguments.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True,
        help="S3 Bucket to use")
    p.add_argument("--infile", required=True,
        help="Path in --bucket to use as input file")
    p.add_argument("--outfile", required=True,
        help="Path in --bucket to use as output file (.kea)")
    p.add_argument("--pickle", required=True,
        help="name of pickle with the result of the preparation")
    p.add_argument("--overlapsize", required=True, type=int,
        help="Tile Overlap to use. (default=%(default)s)")
    p.add_argument("--stats", help="path to json file specifying stats in format:" +
        "bucket:path/in/bucket.json")
    p.add_argument("--nogdalstats", action="store_true", default=False,
        help="don't calculate GDAL's statistics or write a colour table. " + 
            "Can't be used with --stats.")

    cmdargs = p.parse_args()

    return cmdargs


def main():
    """
    Main routine
    """
    cmdargs = getCmdargs()

    # download the pickled data and unpickle.
    s3 = boto3.client('s3')
    with io.BytesIO() as fileobj:
        s3.download_fileobj(cmdargs.bucket, cmdargs.pickle, fileobj)
        fileobj.seek(0)

        dataFromPickle = pickle.load(fileobj)

    # work out GDAL path to input file and open it
    inPath = '/vsis3/' + cmdargs.bucket + '/' + cmdargs.infile
    inDs = gdal.Open(inPath)

    tempDir = tempfile.mkdtemp()

    # work out what the tiles would have been named
    # Note: this needs to match do_tile.py.
    tileFilenames = {}
    for col, row in dataFromPickle['colRowList']:
        filename = '/vsis3/' + cmdargs.bucket + '/' + 'tile_{}_{}.{}'.format(col, row, 'tif')
        tileFilenames[(col, row)] = filename    

    # save the KEA file to the local path first
    localOutfile = os.path.join(tempDir, os.path.basename(cmdargs.outfile))

    # do the stitching. Note maxSegId and hasEmptySegments not used here
    # but ideally they would be saved somewhere also.
    (maxSegId, hasEmptySegments) = tiling.doTiledShepherdSegmentation_finalize(
        inDs, localOutfile, tileFilenames, dataFromPickle['tileInfo'], 
        cmdargs.overlapsize, tempDir)

    # open for the creation of stats
    localDs = gdal.Open(localOutfile, gdal.GA_Update)

    if not cmdargs.nogdalstats:
        # need the histogram for stats
        hist = tiling.calcHistogramTiled(localDs, maxSegId, writeToRat=True)

        band = localDs.GetRasterBand(1)
        utils.estimateStatsFromHisto(band, hist)
        utils.writeRandomColourTable(band, maxSegId + 1)

    # now do any stats the user has asked for
    if cmdargs.stats is not None:

        bucket, stats = cmdargs.stats.split(':')
        with io.BytesIO() as fileobj:
            s3.download_fileobj(bucket, stats, fileobj)
            fileobj.seek(0)

            dataForStats = json.load(fileobj)
            for img, bandnum, selection in dataForStats:
                tilingstats.calcPerSegmentStatsTiled(img, bandnum, 
                    localDs, selection)

    # ensure closed before uploading
    del localDs

    # upload the KEA file
    s3.upload_file(localOutfile, cmdargs.bucket, cmdargs.outfile)

    # cleanup temp files from S3
    objs = [{'Key': cmdargs.pickle}]
    for col, row in tileFilenames:
        filename = 'tile_{}_{}.{}'.format(col, row, 'tif')
        objs.append({'Key': filename})

    s3.delete_objects(Bucket=cmdargs.bucket, Delete={'Objects': objs})

    # cleanup
    shutil.rmtree(tempDir)
    maxMem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print('Max Mem Usage', maxMem)


if __name__ == '__main__':
    main()
