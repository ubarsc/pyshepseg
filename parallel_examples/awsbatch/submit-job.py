#!/usr/bin/env python3

"""
Script that starts a tiled segmentation using AWS Batch.

Submits a job that runs that runs the do_prepare.py
script with the given arguments. 

do_prepare.py then submits an array job to run do_tile.py 
- one job per tile. It also submits a do_stitch.py job that 
is dependent on all the do_tile.py jobs finishing. 
"""

import argparse
import boto3

# name of pickle to save to S3 with the info needed for each tile
PICKLE_NAME = 'pyshepseg_tiling.pkl'


def getCmdargs():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True,
        help="S3 Bucket to use")
    p.add_argument("--infile", required=True,
        help="Path in --bucket to use as input file")
    p.add_argument("--outfile", required=True,
        help="Path in --bucket to use as output file (.kea)")
    p.add_argument("-b", "--bands", default="3,4,5", 
        help="Comma seperated list of bands to use. 1-based. (default=%(default)s)")
    p.add_argument("--jobqueue", default="PyShepSegBatchProcessingJobQueue",
        help="Name of Job Queue to use. (default=%(default)s)")
    p.add_argument("--jobdefnprepare", default="PyShepSegBatchJobDefinitionTile",
        help="Name of Job Definition to use for the preparation job. (default=%(default)s)")
    p.add_argument("--jobdefntile", default="PyShepSegBatchJobDefinitionTile",
        help="Name of Job Definition to use for tile jobs. (default=%(default)s)")
    p.add_argument("--jobdefnstitch", default="PyShepSegBatchJobDefinitionStitch",
        help="Name of Job Definition to use for the stitch job. (default=%(default)s)")
    p.add_argument("--region", default="eu-central-1",
        help="Region to run the jobs in. (default=%(default)s)")
    p.add_argument("--tilesize", default=4096, type=int,
        help="Tile Size to use. (default=%(default)s)")
    p.add_argument("--overlapsize", default=1024, type=int,
        help="Tile Overlap to use. (default=%(default)s)")

    cmdargs = p.parse_args()

    return cmdargs


def main():
    cmdargs = getCmdargs()
    
    batch = boto3.client('batch', region_name=cmdargs.region)

    # submit the prepare job
    response = batch.submit_job(jobName="pyshepseg_prepare",
            jobQueue=cmdargs.jobqueue,
            jobDefinition=cmdargs.jobdefntile,
            containerOverrides={
                "command": ['/usr/bin/python3', '/ubarscsw/bin/do_prepare.py',
                    '--bucket', cmdargs.bucket, '--pickle', PICKLE_NAME,
                    '--infile', cmdargs.infile, '--outfile', cmdargs.outfile,
                    '--bands', cmdargs.bands, '--tilesize', str(cmdargs.tilesize), 
                    '--overlapsize', str(cmdargs.overlapsize),
                    '--jobqueue', cmdargs.jobqueue, 
                    '--jobdefntile', cmdargs.jobdefntile,
                    '--jobdefnstitch', cmdargs.jobdefnstitch]})
    prepareId = response['jobId']
    print('Prepare Job Id', prepareId)
    

if __name__ == '__main__':
    main()
