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


def getCmdargs():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True,
        help="S3 Bucket to use")
    p.add_argument("--infile", required=True,
        help="Path in --bucket to use as input file")
    p.add_argument("--outfile", required=True,
        help="Path in --bucket to use as output file (.kea)")
    p.add_argument("--tileprefix",
        help="Unique prefix to save the output tiles with.")
    p.add_argument("-b", "--bands",
        help="Comma seperated list of bands to use. 1-based. Uses all bands by default.")
    p.add_argument("--jobqueue", default="PyShepSegBatchProcessingJobQueue",
        help="Name of Job Queue to use. (default=%(default)s)")
    p.add_argument("--jobdefnprepare", default="PyShepSegBatchJobDefinitionTile",
        help="Name of Job Definition to use for the preparation job. (default=%(default)s)")
    p.add_argument("--jobdefntile", default="PyShepSegBatchJobDefinitionTile",
        help="Name of Job Definition to use for tile jobs. (default=%(default)s)")
    p.add_argument("--jobdefnstitch", default="PyShepSegBatchJobDefinitionStitch",
        help="Name of Job Definition to use for the stitch job. (default=%(default)s)")
    p.add_argument("--region", default="us-west-2",
        help="Region to run the jobs in. (default=%(default)s)")
    p.add_argument("--tilesize", default=4096, type=int,
        help="Tile Size to use. (default=%(default)s)")
    p.add_argument("--overlapsize", default=1024, type=int,
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
            "Can't be used with --stats or --spatialstats.")
    p.add_argument("--minSegmentSize", type=int, default=50, required=False,
        help="Segment size for segmentation (default=%(default)s)")
    p.add_argument("--numClusters", type=int, default=60, required=False,
        help="Number of clusters for segmentation (default=%(default)s)")
    p.add_argument("--maxSpectDiff", required=False, default='auto',
        help="Maximum spectral difference for segmentation (default=%(default)s)")
    p.add_argument("--spectDistPcntile", type=int, default=50, required=False,
        help="Spectral Distance Percentile for segmentation (default=%(default)s)")

    cmdargs = p.parse_args()

    return cmdargs


def main():
    cmdargs = getCmdargs()
    
    batch = boto3.client('batch', region_name=cmdargs.region)

    pickleName = 'pyshepseg_tiling.pkl'
    # make unique also if tiles are
    if cmdargs.tileprefix is not None:
        pickleName = '{}_pyshepseg_tiling.pkl'.format(cmdargs.tileprefix)

    cmd = ['/usr/bin/python3', '/ubarscsw/bin/do_prepare.py',
        '--region', cmdargs.region,
        '--bucket', cmdargs.bucket, '--pickle', pickleName,
        '--infile', cmdargs.infile, '--outfile', cmdargs.outfile,
        '--tilesize', str(cmdargs.tilesize), 
        '--overlapsize', str(cmdargs.overlapsize),
        '--jobqueue', cmdargs.jobqueue, 
        '--jobdefntile', cmdargs.jobdefntile,
        '--jobdefnstitch', cmdargs.jobdefnstitch,
        '--minSegmentSize', str(cmdargs.minSegmentSize),
        '--numClusters', str(cmdargs.numClusters),
        '--maxSpectDiff', cmdargs.maxSpectDiff,
        '--spectDistPcntile', str(cmdargs.spectDistPcntile)]
    if cmdargs.bands is not None:
        cmd.extend(['--bands', cmdargs.bands])
    if cmdargs.stats is not None:
        cmd.extend(['--stats', cmdargs.stats])
    if cmdargs.spatialstats is not None:
        cmd.extend(['--spatialstats', cmdargs.spatialstats])
    if cmdargs.nogdalstats:
        cmd.append('--nogdalstats')
    if cmdargs.tileprefix is not None:
        cmd.extend(['--tileprefix', cmdargs.tileprefix])

    # submit the prepare job
    response = batch.submit_job(jobName="pyshepseg_prepare",
            jobQueue=cmdargs.jobqueue,
            jobDefinition=cmdargs.jobdefntile,
            containerOverrides={
                "command": cmd})
    prepareId = response['jobId']
    print('Prepare Job Id', prepareId)
    

if __name__ == '__main__':
    main()
