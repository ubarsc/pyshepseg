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
    p.add_argument("--tileprefix", required=True,
        help="Unique prefix to save the output tiles with.")
    p.add_argument("--pickle", required=True,
        help="name of pickle with the result of the preparation")
    p.add_argument("--overlapsize", required=True, type=int,
        help="Tile Overlap to use. (default=%(default)s)")
    p.add_argument("--stats", help="path to json file specifying stats in format:" +
        "bucket:path/in/bucket.json. Contents must be a list of [img, band, " +
        "statsSelection] tuples.")
    p.add_argument("--spatialstats", help="path to json file specifying spatial " +
        "stats in format: bucket:path/in/bucket.jso. Contents must be a list of " +
        "[img, band, [list of (colName, colType) tuples], name-of-userfunc, param]" +
        " tuples.")
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
        filename = '/vsis3/' + cmdargs.bucket + '/' + '{}_{}_{}.{}'.format(
            cmdargs.tileprefix, col, row, 'tif')
        tileFilenames[(col, row)] = filename    

    # save the KEA file to the local path first
    localOutfile = os.path.join(tempDir, os.path.basename(cmdargs.outfile))

    # do the stitching. Note maxSegId and hasEmptySegments not used here
    # but ideally they would be saved somewhere also.
    (maxSegId, hasEmptySegments, localDs) = tiling.doTiledShepherdSegmentation_finalize(
        inDs, localOutfile, tileFilenames, dataFromPickle['tileInfo'], 
        cmdargs.overlapsize, tempDir, True)

    # clean up files to release space
    objs = []
    for col, row in tileFilenames:
        filename = '{}_{}_{}.{}'.format(cmdargs.tileprefix, col, row, 'tif')
        objs.append({'Key': filename})

    # workaround 1000 at a time limit
    while len(objs) > 0:
        s3.delete_objects(Bucket=cmdargs.bucket, Delete={'Objects': objs[0:1000]})
        del objs[0:1000]

    if not cmdargs.nogdalstats:
        band = localDs.GetRasterBand(1)
        utils.estimateStatsFromHisto(band, hist)
        utils.writeRandomColourTable(band, maxSegId + 1)
        utils.addOverviews(localDs)

    # now do any stats the user has asked for
    if cmdargs.stats is not None:

        bucket, statsKey = cmdargs.stats.split(':')
        with io.BytesIO() as fileobj:
            s3.download_fileobj(bucket, statsKey, fileobj)
            fileobj.seek(0)

            dataForStats = json.load(fileobj)
            for img, bandnum, selection in dataForStats:
                print(img, bandnum, selection)
                tilingstats.calcPerSegmentStatsTiled(img, bandnum, 
                    localDs, selection)

    if cmdargs.spatialstats is not None:
        bucket, spatialstatsKey = cmdargs.spatialstats.split(':')
        with io.BytesIO() as fileobj:
            s3.download_fileobj(bucket, spatialstatsKey, fileobj)
            fileobj.seek(0)

            dataForStats = json.load(fileobj)
            for img, bandnum, colInfo, userFuncName, param in dataForStats:
                print(img, bandnum, colInfo, userFuncName, param)
                if (not userFuncName.startswith('userFunc') or
                        not hasattr(tilingstats, userFuncName)):
                    raise ValueError("'userFunc' must be a function starting " +
                        "with 'userFunc' in the tilingstats module")

                userFunc = getattr(tilingstats, userFuncName)
                tilingstats.calcPerSegmentSpatialStatsTiled(img, bandnum, 
                    localDs, colInfo, userFunc, param)

    # ensure closed before uploading
    del localDs

    # upload the KEA file
    s3.upload_file(localOutfile, cmdargs.bucket, cmdargs.outfile)

    # cleanup temp files from S3
    objs = [{'Key': cmdargs.pickle}]
    if cmdargs.stats is not None:
        objs.append({'Key': statsKey})
    if cmdargs.spatialstats is not None:
        objs.append({'Key': spatialstatsKey})

    s3.delete_objects(Bucket=cmdargs.bucket, Delete={'Objects': objs})

    # cleanup
    shutil.rmtree(tempDir)
    maxMem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print('Max Mem Usage', maxMem)


if __name__ == '__main__':
    main()
