#!/bin/bash

if [[ -z "${AWS_REGION}" ]]; then
    echo "Must set AWS_REGION first"
    exit 1
fi

aws cloudformation delete-stack --stack-name pyshepseg-parallel --region $AWS_REGION
echo 'Stack Deletion in progress... Wait a few minutes'
