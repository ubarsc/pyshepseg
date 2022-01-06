#!/bin/bash

aws cloudformation delete-stack --stack-name ubarsc-parallel-seg --region eu-central-1
echo 'Stack Deletion in progress... Wait a few minutes'
