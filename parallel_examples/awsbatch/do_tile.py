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

# set by AWS Batch
ARRAY_INDEX = os.getenv('AWS_BATCH_JOB_ARRAY_INDEX')
if ARRAY_INDEX is None:
    raise SystemExit('Must set AWS_BATCH_JOB_ARRAY_INDEX env var')

ARRAY_INDEX = int(ARRAY_INDEX)

def getCmdargs():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True,
        help="S3 Bucket to use")
    p.add_argument("--infile", required=True,
        help="Path in --bucket to use as input file")
    p.add_argument("--pickle", required=True,
        help="name of pickle with the result of the preparation")

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

    col, row = dataFromPickle['colRowList'][ARRAY_INDEX]

    filename = 'tile_{}_{}.{}'.format(col, row, 'tif')
    filename = os.path.join(tempDir, filename)

    segResult = tiling.doTiledShepherdSegmentation_doOne(inDs, filename,
            dataFromPickle['tileInfo'], col, row, dataFromPickle['bandNumbers'],
            dataFromPickle['imgNullVal'], dataFromPickle['kmeansObj'], 
            tempfilesDriver='GTiff', tempfilesCreationOptions=['COMPRESS=DEFLATE',
                'ZLEVEL=1', 'PREDICTOR=2', 'TILED=YES', 'INTERLEAVE=BAND', 
                'BIGTIFF=NO', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512'])

    s3.upload_file(filename, cmdargs.bucket, os.path.basename(filename))

    shutil.rmtree(tempDir)
    
if __name__ == '__main__':
    main()
