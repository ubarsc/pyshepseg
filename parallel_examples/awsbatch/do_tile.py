#!/usr/bin/env python3

"""
Process an individual tile as part of a tiled segmentation. Indexes
into the pickled colRowList with the AWS_BATCH_JOB_ARRAY_INDEX env var
(set by AWS Batch for array jobs).

"""

import io
import os
import pickle
import argparse
import tempfile
import resource
import shutil
import boto3
from pyshepseg import tiling

from osgeo import gdal

gdal.UseExceptions()

# set by AWS Batch
ARRAY_INDEX = os.getenv('AWS_BATCH_JOB_ARRAY_INDEX')
if ARRAY_INDEX is None:
    raise SystemExit('Must set AWS_BATCH_JOB_ARRAY_INDEX env var')

ARRAY_INDEX = int(ARRAY_INDEX)


def getCmdargs():
    """
    Process the command line arguments.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True,
        help="S3 Bucket to use")
    p.add_argument("--infile", required=True,
        help="Path in --bucket to use as input file")
    p.add_argument("--tileprefix", required=True,
        help="Unique prefix to save the output tiles with.")
    p.add_argument("--pickle", required=True,
        help="name of pickle with the result of the preparation")
    p.add_argument("--minSegmentSize", type=int, default=50, required=False,
        help="Segment size for segmentation (default=%(default)s)")
    p.add_argument("--maxSpectDiff", required=False, default='auto',
        help="Maximum spectral difference for segmentation (default=%(default)s)")
    p.add_argument("--spectDistPcntile", type=int, default=50, required=False,
        help="Spectral Distance Percentile for segmentation (default=%(default)s)")

    cmdargs = p.parse_args()

    return cmdargs


def main():
    """
    Main routine
    """
    cmdargs = getCmdargs()

    # download pickle file and un-pickle it
    s3 = boto3.client('s3')
    with io.BytesIO() as fileobj:
        s3.download_fileobj(cmdargs.bucket, cmdargs.pickle, fileobj)
        fileobj.seek(0)

        dataFromPickle = pickle.load(fileobj)

    # work out GDAL path to input file and open it
    inPath = '/vsis3/' + cmdargs.bucket + '/' + cmdargs.infile
    inDs = gdal.Open(inPath)

    tempDir = tempfile.mkdtemp()

    # work out which tile we are processing
    col, row = dataFromPickle['colRowList'][ARRAY_INDEX]

    # work out a filename to save with the output of this tile
    # Note: this filename format is repeated in do_stitch.py
    # - they must match. Potentially a database or similar
    # could have been used to notify of the names of tiles 
    # but this would add more complexity.
    filename = '{}_{}_{}.{}'.format(cmdargs.tileprefix, 
        col, row, 'tif')
    filename = os.path.join(tempDir, filename)

    # test if int
    maxSpectDiff = cmdargs.maxSpectDiff
    if maxSpectDiff != 'auto':
        maxSpectDiff = int(maxSpectDiff)

    # run the segmentation on this tile.
    # save the result as a GTiff so do_stitch.py can open this tile
    # directly from S3.
    # TODO: create COG instead
    tiling.doTiledShepherdSegmentation_doOne(inDs, filename,
        dataFromPickle['tileInfo'], col, row, dataFromPickle['bandNumbers'],
        dataFromPickle['imgNullVal'], dataFromPickle['kmeansObj'], 
        minSegmentSize=cmdargs.minSegmentSize,
        spectDistPcntile=cmdargs.spectDistPcntile, maxSpectralDiff=maxSpectDiff,
        tempfilesDriver='GTiff', tempfilesCreationOptions=['COMPRESS=DEFLATE',
        'ZLEVEL=1', 'PREDICTOR=2', 'TILED=YES', 'INTERLEAVE=BAND', 
        'BIGTIFF=NO', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512'])

    # upload the tile to S3.
    s3.upload_file(filename, cmdargs.bucket, os.path.basename(filename))

    # cleanup
    shutil.rmtree(tempDir)
    maxMem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print('Max Mem Usage', maxMem)


if __name__ == '__main__':
    main()
