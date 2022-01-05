#!/usr/bin/env python3

import io
import pickle
import argparse
import boto3
from pyshepseg import tiling

def getCmdargs():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True,
        help="S3 Bucket to use")
    p.add_argument("--infile", required=True,
        help="Path in --bucket to use as input file")
    p.add_argument("-b", "--bands", default="3,4,5", 
        help="Comma seperated list of bands to use. 1-based. (default=%(default)s)")
    p.add_argument("--tilesize", required=True, type=int,
        help="Tile Size to use. (default=%(default)s)")
    p.add_argument("--overlapsize", required=True, type=int,
        help="Tile Overlap to use. (default=%(default)s)")
    p.add_argument("--pickle", required=True,
        help="name of pickle to save result of preparation into")
    p.add_argument("--region", default="eu-central-1",
        help="Region to run the jobs in. (default=%(default)s)")
    p.add_argument("--jobqueue", default="PyShepSegBatchProcessingJobQueue",
        help="Name of Job Queue to use. (default=%(default)s)")
    p.add_argument("--jobdefntile", default="PyShepSegBatchJobDefinitionTile",
        help="Name of Job Definition to use for tile jobs. (default=%(default)s)")
    p.add_argument("--jobdefnstitch", default="PyShepSegBatchJobDefinitionStitch",
        help="Name of Job Definition to use for the stitch job. (default=%(default)s)")

    cmdargs = p.parse_args()
    # turn string of bands into list of ints
    cmdargs.bands = [int(x) for x in cmdargs.bands.split(',')]

    return cmdargs

def main():
    cmdargs = getCmdargs()

    batch = boto3.client('batch', region_name=cmdargs.region)
    s3 = boto3.client('s3')

    inPath = '/vsis3/' + cmdargs.bucket + '/' + cmdargs.infile

    # run the initial part of the tiled segmentation
    inDs, bandNumbers, kmeansObj, subsamplePcnt, imgNullVal, tileInfo = (
        tiling.doTiledShepherdSegmentation_prepare(inPath, 
        bandNumbers=cmdargs.bands, tileSize=cmdargs.tilesize, 
        overlapSize=cmdargs.overlapsize))

    # pickle the required input data that each of the tiles will need
    colRowList = sorted(tileInfo.tiles.keys(), key=lambda x:(x[1], x[0]))
    dataToPickle = {'tileInfo': tileInfo, 'colRowList': colRowList, 
        'bandNumbers': bandNumbers, 'imgNullVal': imgNullVal, 
        'kmeansObj': kmeansObj}
    with io.BytesIO() as fileobj:
        pickle.dump(dataToPickle, fileobj)
        fileobj.seek(0)
        s3.upload_fileobj(fileobj, cmdargs.bucket, cmdargs.pickle)

    # now submit an array job with all the tiles
    response = batch.submit_job(jobName="pyshepseg_tiles",
            jobQueue=cmdargs.jobqueue,
            jobDefinition=cmdargs.jobdefntile,
            arrayProperties={'size': len(colRowList)},
            containerOverrides={
                "command": ['/usr/bin/python3', '/ubarscsw/bin/do_tile.py',
                    '--bucket', cmdargs.bucket, '--pickle', cmdargs.pickle,
                    '--infile', cmdargs.infile]})
    tilesJobId = response['jobId']
    print('Tiles Job Id', tilesJobId)

    # now submit a dependent job with the stitching
    """
    response = batch.submit_job(jobName="pyshepseg_stitch",
            jobQueue=cmdargs.jobqueue,
            jobDefinition=cmdargs.jobdefnstitch,

            dependsOn=[{'jobId': tilesJobId}],
            containerOverrides={
                "command": ['/usr/bin/python3', '/ubarscsw/bin/do_stitch.py',
                    '--bucket', cmdargs.bucket, '--outfile', cmdargs.outfile,
                    '--infile', cmdargs.infile, '--pickle', cmdargs.pickle,
                    '--tilesize', str(cmdargs.tilesize), 
                    '--overlapsize', str(cmdargs.overlapsize)]})
    print('Stitching Job Id', response['jobId'])
    """

if __name__ == '__main__':
    main()
