import xrayspectrapy as xsp
import os
import math
import numpy as np
import random

import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array

def main():
    directory = '~/work/all_rfdata_unique/'
    filedir = os.path.expanduser(directory)
    calcFiles = [os.path.join(filedir, f) for f in os.listdir(filedir)
                   if os.path.isfile(os.path.join(filedir, f))]
                   # if ('CalcInAs' in f or 'calc' in f)]
    
    # output 100 random images
    #img = xsp.datadefs.image.fromFile(calcFiles[0])
    #img = xsp.pdf.normalize_image(img)
    #stdev = 0.06
    #mean = 0.004
    #n = 100
    #output = [img.distances]
    #for i in range(0, n):
    #    randFn = lambda x: xsp.pdf.rand_normal_peaks(x, mean, stdev, 7, 14)
    #    noiseIm = xsp.pdf.noisify_image(img, randFn, 0.004)
    #    output.append(noiseIm.frequencies)
    #print('\n'.join(['\t'.join([str(x) for x in out]) for out in output]))

    calcImages = [xsp.datadefs.image.fromFile(f) for f in calcFiles]
    calcSmoothedImages = [xsp.pdf.smooth_image(im, 0.004) for im in calcImages]
    calcSmoothedImages = [xsp.pdf.normalize_image(im) for im in calcSmoothedImages]

    vals = [(0, 0),
            (0.0015, 0),
            (0.0036, 0),
            (0.0036, 0.003),
            (0.0036, 0.01),
            (0.0036, 0.02),
            (0.0036, 0.03),
            (0.0036, 0.04),
            (0.0036, 0.05),
            (0.0036, 0.06)]

    n = 50
    for val in vals:
        accuracy = getAccuracy(calcImages, calcSmoothedImages, n, val[0],
                                    val[1])
        print("Mean: " + str(val[0]) + ', StDev: ' + str(val[1]) + \
                ', Accuracy: ' + str(accuracy))

def getAccuracy(calcImages, calcSmoothedImages, n, mean, stdev):
    success = 0
    for i in range(0, n):
        randIndex = random.randrange(0, len(calcImages))
        randImg = calcImages[randIndex]
        randFn = lambda x: xsp.pdf.rand_normal_peaks(x, mean, stdev, 7, 14)
        exprImg = xsp.pdf.noisify_image(randImg, randFn, 0.004)

        dists = [xsp.comparers.l2_norm(exprImg, calcImg)
                 for calcImg in calcSmoothedImages]
        if np.argmin(dists) == randIndex:
            success = success + 1
    
    return success / n

main()
