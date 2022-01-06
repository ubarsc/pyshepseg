#!/bin/sh

aws cloudformation create-stack --stack-name ubarsc-parallel-seg \
    --template-body file://template/template.yaml \
    --capabilities CAPABILITY_NAMED_IAM --region eu-central-1
