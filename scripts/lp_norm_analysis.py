import xrayspectrapy as xsp
import math
import numpy as np
import script_utils as su
import os
import sys
import matplotlib as mpl
import random

def runSynthExptRecognitionAnalysis(distFun):
    nSamples = 500
    tSmooth = 0.0092
    direc = os.path.expanduser('~/work/all_rfdata_unique/')
    unSmoothImages = su.getAllImages(direc, ['Calc', 'calc'])
    smoothImages = xsp.pdf.smooth_images(unSmoothImages, tSmooth)
    smoothImages = [xsp.pdf.normalize_image(im) for im in smoothImages]

    mean = 0.004
    stdevs = [0, 0.003, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    for stdev in stdevs:
        randomNoiseFn = lambda x: xsp.pdf.rand_normal_peaks(x, mean, stdev, 7, 14)
        accuracy = getSynthExptAccuracy(nSamples, unSmoothImages, smoothImages,
                distFun, randomNoiseFn, tSmooth)
        print(str(mean) + "," + str(stdev) + "," + str(accuracy))

def getSynthExptAccuracy(nSamples, unSmoothImages, smoothImages,  
        distFun, randomNoiseFn, tSmooth):
    success = 0
    for i in range(0, nSamples):
        idx = random.randrange(0, len(unSmoothImages))
        randImg = unSmoothImages[idx]
        exprImg = xsp.pdf.noisify_image(randImg, randomNoiseFn, tSmooth)
        matchIdx = findIndexOfMatch(smoothImages, exprImg, distFun)
        if idx == matchIdx:
            success = success + 1

    return success / nSamples

def findIndexOfMatch(smoothImages, targetImage, distFun):
    dists = [distFun(targetImage, calcImg) for calcImg in smoothImages]
    return np.argmin(dists)

def getMatches(distFun):
    (exptImages, calcImages) = getAllImages()

    maxClosest = 2
    allOutput = []
    labels = [im.label for im in calcImages]
    for exptImage in exptImages:
        dists = [distFun(exptImage, calcImg) for calcImg in calcImages]
        exptOutput = [exptImage.label]
        exptOutput.extend([labels[np.argsort(dists)[i]] 
                            for i in range(0,maxClosest)])
        allOutput.append(exptOutput)

    return allOutput

def printMatches(matches):
    headerRow = ['Expt']
    headerRow.extend('%d' % (i+1) for i in range(0, len(matches[0])-1))
    allOutput = [headerRow]
    allOutput.extend(matches)
    print(su.matToStr(allOutput))

def plotL1Matches():
    (exptImages, calcImages) = getAllImages()
    images = [im for sublist in [exptImages, calcImages] for im in sublist]

    bestMatches = [
            ['SiLiExpt3',    'SiLiCalc10001'],
            ['SiLiExpt5',    'SiLiCalc11099'],
            ['SiLiExpt6',    'SiLiCalc11099'],
            ['ExptInAs',    'SiLiCalc11780'],
            ['SiLiExpt7',    'SiLiCalc10482'],
            ['SiLiExpt8',    'SiLiCalc10482'],
            ['ExptGaAs',    'CalcGaAs'],
            ['SiLiExpt2',    'SiLiCalc10001'],
            ['SiLiExpt4',    'SiLiCalc10003'],
            ['SiLiExpt1',    'SiLiCalc10001']]
    
    for labels in bestMatches:
        plotImages(labels, images, 'L1Norm')

def plotL2Matches():
    (exptImages, calcImages) = getAllImages()
    images = [im for sublist in [exptImages, calcImages] for im in sublist]

    bestMatches = [
            ['SiLiExpt3', 'SiLiCalc10001'],
            ['SiLiExpt5', 'SiLiCalc10616'],
            ['SiLiExpt6', 'SiLiCalc10616'],
            ['ExptInAs',  'SiLiCalc10429'],
            ['SiLiExpt7', 'SiLiCalc10616'],
            ['SiLiExpt8', 'SiLiCalc10693'],
            ['ExptGaAs',  'CalcGaAs'],
            ['SiLiExpt2', 'SiLiCalc10001'],
            ['SiLiExpt4', 'SiLiCalc10003'],
            ['SiLiExpt1', 'SiLiCalc10001']]
    
    for labels in bestMatches:
        plotImages(labels, images, 'L2Norm')

def plotImages(labels, images, prefix):
    selectImages = [im for im in images if any([im.label in s for s in labels])]
    su.plotImages('~/work/final_figs/', selectImages, prefix, '.png',
            legendLoc=1)

def getAllImages():
    filedir = os.path.expanduser('~/work/all_rfdata_unique/')
    exptImages = su.getAllImages(filedir, ['Expt'])
    calcImages = su.getAllImages(filedir, ['Calc'])
    calcImages = xsp.pdf.smooth_images(calcImages, 0.0092)
    calcImages = [xsp.pdf.normalize_image(im) for im in calcImages]
    return (exptImages, calcImages)

def chooseDistFun(index):
    if index == 1:
        return xsp.comparers.l1_norm
    elif index == 2:
        return xsp.comparers.l2_norm
    else:
        raise ValueError('unknown index')

mpl.rcParams['font.size'] = 20
# printMatches(getMatches(xsp.comparers.l2_norm))
# plotL2Matches()
runSynthExptRecognitionAnalysis(chooseDistFun(int(sys.argv[1])))

