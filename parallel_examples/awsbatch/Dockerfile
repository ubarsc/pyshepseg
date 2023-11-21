FROM ubuntu:21.04

# Needed in case tzdata gets upgraded
ENV TZ=Australia/Brisbane
ARG DEBIAN_FRONTEND=noninteractive

ARG AWS_REGION=us-west-2
ENV AWS_REGION_ENV=$AWS_REGION

# use local mirror
RUN sed -i "s/http:\/\/ports./http:\/\/${AWS_REGION_ENV}.ec2.ports./g" /etc/apt/sources.list
RUN sed -i "s/http:\/\/archive./http:\/\/${AWS_REGION_ENV}.ec2.archive./g" /etc/apt/sources.list

# Update Ubuntu software stack and install base GDAL stack
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y python3-gdal python3-boto3 python3-setuptools python3-sklearn \
		python3-numba wget g++ cmake libhdf5-dev libgdal-dev python3-pip

ENV SW_VOLUME=/ubarscsw
RUN mkdir $SW_VOLUME

ENV SERVICEUSER=ubarscuser
RUN useradd --create-home --shell /bin/bash ${SERVICEUSER}

ENV KEALIB_VERSION=1.5.2
RUN cd /tmp \
    && wget -q https://github.com/ubarsc/kealib/releases/download/kealib-${KEALIB_VERSION}/kealib-${KEALIB_VERSION}.tar.gz \
    && tar xf kealib-${KEALIB_VERSION}.tar.gz \
    && cd kealib-${KEALIB_VERSION} \
    && mkdir build \
    && cd build \
    && cmake -D CMAKE_INSTALL_PREFIX=${SW_VOLUME} -D LIBKEA_WITH_GDAL=ON .. \
    && make \
    && make install \
    && cd ../.. \
    && rm -rf kealib-${KEALIB_VERSION} kealib-${KEALIB_VERSION}.tar.gz

COPY pyshepseg-$PYSHEPSEG_VER.tar.gz /tmp
# install RIOS
RUN cd /tmp && tar xf pyshepseg-$PYSHEPSEG_VER.tar.gz \
    && cd pyshepseg-$PYSHEPSEG_VER \
    && pip install . \
    && cd .. && rm -rf pyshepseg-$PYSHEPSEG_VER pyshepseg-$PYSHEPSEG_VER.tar.gz


ENV PYTHONPATH=${SW_VOLUME}/lib/python3.9/site-packages
ENV LD_LIBRARY_PATH=${SW_VOLUME}/lib
ENV GDAL_DRIVER_PATH=${SW_VOLUME}/lib/gdalplugins

ENV PYTHONUNBUFFERED=1

ENV GDAL_PAM_ENABLED=NO
ENV GDAL_CACHEMAX=1024000000
ENV GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR
ENV GDAL_HTTP_MERGE_CONSECUTIVE_RANGES=YES
ENV GDAL_HTTP_MULTIPLEX=YES
ENV CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff,.vrt,.zip"
ENV VSI_CACHE=True
ENV VSI_CACHE_SIZE=1024000000
ENV GDAL_HTTP_MAX_RETRY=10
ENV GDAL_HTTP_MAX_RETRY=3
ENV CPL_ZIP_ENCODING=UTF-8

COPY do_prepare.py $SW_VOLUME/bin
COPY do_tile.py $SW_VOLUME/bin
COPY do_stitch.py $SW_VOLUME/bin

RUN apt-get remove wget g++ cmake
RUN apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

USER $SERVICEUSER

# a few quick tests
#RUN gdal_translate --formats | grep KEA
RUN python3 -c 'from osgeo import gdal;assert(gdal.GetDriverByName("KEA") is not None)'
RUN python3 -c 'from pyshepseg import tiling'

# export the volume
VOLUME $SW_VOLUME

# set the workdir to the home directory for our user (not sure if right thing to
WORKDIR /home/${SERVICEUSER}
