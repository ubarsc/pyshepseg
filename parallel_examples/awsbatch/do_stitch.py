#!/usr/bin/env python3


import io
import os
import pickle
import argparse
import tempfile
import shutil
import boto3
from pyshepseg import tiling
from osgeo import gdal

def getCmdargs():
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

    cmdargs = p.parse_args()

    return cmdargs

def main():
    cmdargs = getCmdargs()

    s3 = boto3.client('s3')
    with io.BytesIO() as fileobj:
        s3.download_fileobj(cmdargs.bucket, cmdargs.pickle, fileobj)
        fileobj.seek(0)

        dataFromPickle = pickle.load(fileobj)

    inPath = '/vsis3/' + cmdargs.bucket + '/' + cmdargs.infile
    inDs = gdal.Open(inPath)

    tempDir = tempfile.mkdtemp()

    # work out what the tiles would have been named
    tileFilenames = {}
    for col, row in dataFromPickle['colRowList']:
        filename = '/vsis3/' + cmdargs.bucket + '/' + 'tile_{}_{}.{}'.format(col, row, 'tif')
        tileFilenames[(col, row)] = filename    

    localOutfile = os.path.join(tempDir, os.path.basename(cmdargs.outfile))

    maxSegId, hasEmptySegments = tiling.doTiledShepherdSegmentation_finalize(
            inDs, tempDir, tileFilenames, dataFromPickle['tileInfo'], 
            cmdargs.overlapsize, tempDir)

    s3.upload_file(localOutfile, cmdargs.bucket, cmdargs.outfile)

    objs = [{'Key': cmdargs.pickle}]
    for col, row in tileFilenames:
        obj.append({'Key': tileFilenames[(col, row)]})

    s3.delete_objects(cmdargs.bucket, {'Objects': objs})

    shutil.rmtree(tempDir)

if __name__ == '__main__':
    main()
