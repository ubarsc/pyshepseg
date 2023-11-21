#!/bin/sh

if [[ -z "${AWS_REGION}" ]]; then
    echo "Must set AWS_REGION first"
    exit 1
fi

aws cloudformation create-stack --stack-name pyshepseg-parallel \
    --template-body file://template/template.yaml \
    --capabilities CAPABILITY_NAMED_IAM --region $AWS_REGION \
    --tags Key=PyShepSeg,Value=1
