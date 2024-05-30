# A make file to create and push a docker image with RIOS and any other required packages
# to ECR.
# set the AWS_REGION environment variable to the name of the AWS region you wish to use

ifndef AWS_REGION
$(error AWS_REGION is not set)
endif

ACCOUNT_ID := $(shell aws sts get-caller-identity --query "Account" --output text)
PYSHEPSEG_VER := $(shell python3 -c 'import pyshepseg;print(pyshepseg.__version__)')
ECR_URL=${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
DOCKER_TAG=pyshepseg

REPO=${ECR_URL}/${DOCKER_TAG}:latest

default: all

# grab the current pyshepseg source tree and make it available to the 
# docker COPY command
dist:
	cd ../../;python3 -m build .
	cp ../../dist/pyshepseg-$(PYSHEPSEG_VER).tar.gz .

# Login to ECR, build package and push to ECR
all: dist
	aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin $(ECR_URL)
	docker build --build-arg AWS_REGION=${AWS_REGION} --build-arg PYSHEPSEG_VER=$(PYSHEPSEG_VER) -t $(DOCKER_TAG) .
	docker tag $(DOCKER_TAG) $(REPO)
	docker push $(REPO)
