import sys, os
parentdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parentdir)

import xrayspectrapy as xsp

def exportDists():
    filedir = os.path.expanduser('~/work/rfdata/')
    allCalcFiles = [os.path.join(filedir, f) for f in os.listdir(filedir)
                   if os.path.isfile(os.path.join(filedir, f))
                   if ('Calc' in f or 'calc' in f)]

    allCalcImages = [xsp.datadefs.image.fromFile(f) for f in allCalcFiles]

    allCalcImagesSub = allCalcImages

    for im1 in allCalcImagesSub:
        for im2 in allCalcImagesSub:
            if (im1.label < im2.label):
                dist = xsp.comparers.l2_norm(im1, im2)
                print(im1.label + "," + im2.label + "," + "%.12f" %dist)

# filter_calc_data.py > 'data.csv'

def filterDuplicateDists():
    dataFile = open(os.path.expanduser('~/work/data.csv'))

    for line in dataFile:
        lineContents = line.rstrip('\n')
        lineParts = lineContents.split(',')
        if lineParts[0] < lineParts[1]:
            print(lineContents)

def readDists():
    dataFile = open(os.path.expanduser('~/work/data.csv'))

    result = []
    for line in dataFile:
        lineContents = line.rstrip('\n')
        lineParts = lineContents.split(',')
        result.append(float(lineParts[2]))

    dataFile.close()
    return result

def printHistogram():
    import numpy as np
    dists = readDists()
    (freqs, bins) = np.histogram(dists, 50)

    for (f, b) in zip(freqs, bins):
        print(str(f) + '\t' + str(b))

def printSameNames():
    dataFile = open(os.path.expanduser('~/work/data.csv'))

    result = []
    for line in dataFile:
        lineContents = line.rstrip('\n')
        lineParts = lineContents.split(',')
        if float(lineParts[2]) < 0.000000001:
            result.append(lineParts[1])
    
    dataFile.close()

    resultSet = set(result)
    for f in resultSet:
        print(f)

printSameNames()
