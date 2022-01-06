#!/bin/sh

set -e

DOCKER_TAG=ubarsc_parallel_seg
ACCOUNT_ID=`aws sts get-caller-identity --query "Account" --output text`

# ECR
ECR_URL=${ACCOUNT_ID}.dkr.ecr.eu-central-1.amazonaws.com

aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin ${ECR_URL}

docker tag ${DOCKER_TAG} ${ECR_URL}/${DOCKER_TAG}:latest

docker push ${ECR_URL}/${DOCKER_TAG}:latest
