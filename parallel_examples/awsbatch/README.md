# AWS Batch

The files in this folder contain a working demonstration of how
to run the tiled segmentation in parallel on AWS Batch.

## Contents

### CloudFormation

`template/template.yaml`

Contains the CloudFormation template to create the AWS Batch environment

`create-stack.sh`

Invokes CloudFormation to create the AWS Batch environment from `template/template.yaml`
to create a CloudFormation stack called `ubarsc-parallel-seg`.

`delete-stack.sh` 

Deletes the AWS Batch environment.

`modify-stack.sh`

Attempts to modify the AWS Batch environment by applying any changes to `template/template.yaml`.

### Docker

AWS Batch requires a Docker image to be pushed to AWS ECR.

`Dockerfile`

Contains instructions for creating a Docker Image with the required software
to perform the tiled segmentation.

`build_docker.sh`

Builds the Docker Image from Dockerfile

`push_to_ecr.sh`

Pushes the Docker Image to a repository on AWS ECR called "ubarsc_parallel_seg". Note this 
repository is NOT created by the CloudFormation script above.

### Supporting Scripts

These are copied into the Docker Image by the `Dockerfile`.

`do_prepare.py`

Runs `tiling.doTiledShepherdSegmentation_prepare()` and copies the resulting data to a pickle file
to the specified S3 Bucket to be picked up by the following steps.

This is the first time we know how many tiles there are so this script also kicks off
the appropriate number of array jobs (each running `do_tile.py` - see below). It also submits a final
job (dependent on all the `do_tile.py` jobs completing) that runs `do_stitch.py` (see below).

`do_tile.py`

Runs `tiling.doTiledShepherdSegmentation_doOne()`. It loads the required data from the saved pickle
and outputs the processed tile and saves it to the specified S3 Bucket to be picked up by `do_stitch.py`.
Which tile is being processed is specified by the `AWS_BATCH_JOB_ARRAY_INDEX` environment variable -
refer to the AWS Batch documentation on array jobs for more information.

`do_stitch.py`

Runs `tiling.doTiledShepherdSegmentation_finalize()`. It loads the required data from the saved pickle
and determines the names of the individual tiles that have been processed by `do_tile.py`. This
generates the output (stitched) file and copies to S3. It also deletes all temporary files from S3.


