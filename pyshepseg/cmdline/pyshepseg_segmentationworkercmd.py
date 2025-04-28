#!/usr/bin/env python
"""
Main script for a segmentation worker running in a separate process.
"""
import argparse
import queue

import numpy
from osgeo import gdal

from pyshepseg import shepseg
from pyshepseg.tiling import NetworkDataChannel
from pyshepseg.utils import WorkerErrorRecord


# Compute workers in separate processes should always use GDAL exceptions,
# regardless of whether the main script is doing so.
gdal.UseExceptions()


def getCmdargs():
    """
    Get command line arguments
    """
    p = argparse.ArgumentParser(description=("Main script run by each " +
        "segmentation worker"))
    p.add_argument("-i", "--idnum", type=int, help="Worker ID number")
    p.add_argument("--channaddrfile", help="File with data channel address")
    p.add_argument("--channaddr", help=("Directly specified data channel " +
        "address, as 'hostname,portnum,authkey'. This is less secure, and " +
        "should only be used if the preferred option --channaddrfile " +
        "cannot be used"))

    cmdargs = p.parse_args()
    return cmdargs


def mainCmd():
    """
    Main entry point for command script. This is referenced by the install
    configuration to generate the actual command line main script.
    """
    cmdargs = getCmdargs()

    if cmdargs.channaddrfile is not None:
        addrStr = open(cmdargs.channaddrfile).readline().strip()
    else:
        addrStr = cmdargs.channaddr

    (host, port, authkey) = tuple(addrStr.split(','))
    port = int(port)
    authkey = bytes(authkey, 'utf-8')

    pyshepsegRemoteSegmentationWorker(cmdargs.idnum, host, port, authkey)


def pyshepsegRemoteSegmentationWorker(workerID, host, port, authkey):
    """
    The main routine to run a segmentation worker on a remote host.

    """
    dataChan = NetworkDataChannel(hostname=host, portnum=port, authkey=authkey)

    infile = dataChan.segDataDict['infile']
    tileInfo = dataChan.segDataDict['tileInfo']
    minSegmentSize = dataChan.segDataDict['minSegmentSize']
    maxSpectralDiff = dataChan.segDataDict['maxSpectralDiff']
    imgNullVal = dataChan.segDataDict['imgNullVal']
    fourConnected = dataChan.segDataDict['fourConnected']
    kmeansObj = dataChan.segDataDict['kmeansObj']
    verbose = dataChan.segDataDict['verbose']
    spectDistPcntile = dataChan.segDataDict['spectDistPcntile']
    bandNumbers = dataChan.segDataDict['bandNumbers']

    inDs = gdal.Open(infile)

    try:
        colRow = popFromQue(dataChan.inQue)
        while colRow is not None:
            (col, row) = colRow

            xpos, ypos, xsize, ysize = tileInfo.getTile(col, row)

            lyrDataList = []
            for bandNum in bandNumbers:
                with dataChan.readSemaphore:
                    lyr = inDs.GetRasterBand(bandNum)
                    lyrData = lyr.ReadAsArray(xpos, ypos, xsize, ysize)
                    lyrDataList.append(lyrData)

            img = numpy.array(lyrDataList)

            segResult = shepseg.doShepherdSegmentation(img, 
                        minSegmentSize=minSegmentSize,
                        maxSpectralDiff=maxSpectralDiff,
                        imgNullVal=imgNullVal, 
                        fourConnected=fourConnected,
                        kmeansObj=kmeansObj, 
                        verbose=verbose,
                        spectDistPcntile=spectDistPcntile)

            dataChan.segResultCache.addResult(col, row, segResult)
            colRow = popFromQue(dataChan.inQue)
    except Exception as e:
        # Send a printable version of the exception back to main thread
        workerErr = WorkerErrorRecord(e, 'compute', workerID)
        dataChan.exceptionQue.put(workerErr)


def popFromQue(que):
    """
    Pop out the next item from the given Queue, returning None if
    the queue is empty.

    WARNING: don't use this if the queued items can be None
    """
    try:
        item = que.get(block=False)
    except queue.Empty:
        item = None
    return item


if __name__ == "__main__":
    mainCmd()
