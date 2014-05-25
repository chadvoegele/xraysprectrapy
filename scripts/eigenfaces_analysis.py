import xrayspectrapy as xsp
import os
import sys
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import rc

def matToStr(mat):
    return '\n'.join(['\t'.join(['%.8f' % y for y in x]) for x in mat])

def main():
    (_, _, mean, freqMat, eigenvalues, V) = getPCAData(getAllCalcImages())

    (_, freqExpt) = getExptPCAData(mean)
    TExp = np.dot(freqExpt, V)

    # freqMatFromPCs = np.dot(np.dot(freqMat, V), np.transpose(V)) + mean
    print(matToStr(TExp[:,0:2]))

def getExptRecognitionAccuracy():
    nPCs = 20

    (labels, _, mean, freqMat, _, V) = getPCAData(getAllCalcImages())
    T = np.dot(freqMat, V)[:,0:nPCs]

    (exptLabels, freqExpt) = getExptPCAData(mean)
    TExp = np.dot(freqExpt, V)[:,0:nPCs]
    
    output = [[x for x in exptLabels]]
    for i in range(0,5):
        findMatch = lambda x: getIndexOfNthClosestVec(T, x, i+1)
        exptMatches = [labels[findMatch(x)] for x in TExp]
        output.append(exptMatches)

    print('\n'.join(('\t'.join((x for x in row)) for row in output)))

def getSynthExptRecognitionAnalysis(nPCs):
    nSamples = 500
    tSmooth = 0.004
    direc = os.path.expanduser('~/work/all_rfdata_unique/')
    unSmoothImages = getAllImages(direc, ['Calc', 'calc'])
    images = [xsp.pdf.smooth_image(im, tSmooth) for im in unSmoothImages]
    images = [xsp.pdf.normalize_image(im) for im in images]
    (labels, _, mean, freqMat, _, V) = getPCAData(images)
    T = np.dot(freqMat, V)[:,0:nPCs]

    stdevs = [0, 0.003, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    for stdev in stdevs:
        randomNoiseFn = lambda x: xsp.pdf.rand_normal_peaks(x, 0.004, stdev, 7, 14)
        accuracy = getSynthExptAccuracy(nSamples, nPCs, unSmoothImages, T, V,
                mean, randomNoiseFn, tSmooth)
        print(str(stdev) + "," + str(accuracy))

def getSynthExptAccuracy(nSamples, nPCs, unSmoothImages, T, V,
        mean, randomNoiseFn, tSmooth):
    success = 0
    for i in range(0, nSamples):
        idx = random.randrange(0, len(unSmoothImages))
        randImg = unSmoothImages[idx]
        exprImg = xsp.pdf.noisify_image(randImg, randomNoiseFn, tSmooth)
        TExpr = np.dot(exprImg.frequencies - mean, V)[0:nPCs]
        matchIdx = getIndexOfClosestVec(T, TExpr)
        if idx == matchIdx:
            success = success + 1

    return success / nSamples

def getIndexOfClosestVec(vecs, otherVec):
    return np.argmin([np.linalg.norm(vec - otherVec) for vec in vecs])

def getIndexOfNthClosestVec(vecs, otherVec, n):
    # 1st closest: 0 index
    # 2nd closest: 1 index
    return np.argsort([np.linalg.norm(vec - otherVec) for vec in vecs])[n-1]

def printAllCalcFreqs():
    calcImages = getAllCalcImages()
    freqMatMean = np.array([im.frequencies for im in calcImages])
    print(matToStr(freqMatMean))

def plotEigenSpaceScatter():
    (calcLabels, _, mean, freqMat, eigenvalues, V) = getPCAData(getAllCalcImages())
    T = np.dot(freqMat, V)

    (exptLabels, freqExpt) = getExptPCAData(mean)
    TExp = np.dot(freqExpt, V)

    pairs = ((0,1,2),
             (1,2,2),
             (2,3,3),
             (3,4,2))
    pairToStr = lambda x: (str(x[0]+1), str(x[1]+1))

    exptCalcMatches = (('SiLiCalc10001', 'SiLiExpt1'),
                     ('CalcGaAs', 'ExptGaAs'),
                     ('CalcInAs', 'ExptInAs'))

    for pair in pairs:
        plt.plot(T[:,pair[0]], T[:,pair[1]], '.')
        plt.plot(TExp[:,pair[0]], TExp[:,pair[1]], '.r')
        legend=['Calc', 'Expt']

        for (calcLabel, exptLabel) in exptCalcMatches:
            calcIdx = calcLabels.index(calcLabel)
            exptIdx = exptLabels.index(exptLabel)
            plt.plot(T[calcIdx,pair[0]], T[calcIdx,pair[1]], 'o')
            plt.plot(TExp[exptIdx,pair[0]], TExp[exptIdx,pair[1]], '*')
            legend.append(calcLabel)
            legend.append(exptLabel)
        
        plt.legend(legend, ncol=2, loc=pair[2], numpoints=1)
        plt.xlabel('Loading: '+ pairToStr(pair)[0])
        plt.ylabel('Loading: '+ pairToStr(pair)[1])
        directory = os.path.expanduser('~/code/xrayspectrapy/doc/figs')
        filename = 'eigenspace' + pairToStr(pair)[0] + '-' + \
                        pairToStr(pair)[1] + '.eps'
        filenameWPath = os.path.join(directory, filename)
        plt.savefig(filenameWPath, bbox_inches='tight')
        plt.close()

def plotCumulativeVarExplained():
    (_, dists, _, freqMat, eigenvalues, V) = getPCAData(getAllCalcImages())

    explainedByPCs = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    pc = [x+1 for x in range(0, len(explainedByPCs))]
    plt.plot(pc, explainedByPCs, '-')

    plt.axis([0, len(pc)+1, 0, 1.1])
    plt.ylabel('Variance Explained')
    plt.xlabel('Principal Component')
    directory = os.path.expanduser('~/code/xrayspectrapy/doc/figs')
    filename = os.path.join(directory, 'eigenfaces_varexplained.eps')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plotEigenfaces():
    (_, dists, _, freqMat, eigenvalues, V) = getPCAData(getAllCalcImages())

    n = 8
    for i in range(0,n):
        plt.plot(dists, V[:,i], '-')

    plt.axis([1, 8, np.min(V[:,0:n])*0.9, np.max(V[:,0:n])*1.1])
    plt.ylabel('Frequencies')
    plt.xlabel('Distances (Angstroms)')
    filename = os.path.expanduser('~/code/xrayspectrapy/doc/figs/eigenfaces.eps')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plotAllCalcFreqsWMean():
    calcImages = getAllCalcImages()
    dists = calcImages[0].distances
    freqMatMean = np.array([im.frequencies for im in calcImages])
    means = np.mean(freqMatMean, axis=0)

    for freq in freqMatMean:
        plt.plot(dists, freq, '-')

    plt.plot(dists, means, '-', linewidth=7)

    plt.axis([1, 8, 0, np.max(freqMatMean)*1.1])
    plt.ylabel('Frequencies')
    plt.xlabel('Distances (Angstroms)')
    filename = os.path.expanduser('~/work/allIm_mean.png')
    plt.savefig(filename, dpi=250, bbox_inches='tight')
    plt.close()

def plotBestMatches():
    images = [getAllCalcImages(), getAllExptImages()]
    images = [im for sublist in images for im in sublist]

    allLabels = [['SiLiExpt1', 'SiLiCalc10001'],
                 ['SiLiExpt5', 'SiLiCalc10225']]
    
    for labels in allLabels:
        plotImages(labels, images)

def plotImages(labels, images):
    selectImages = [im for im in images if im.label in labels]
    directory = os.path.expanduser('~/code/xrayspectrapy/doc/figs')
    name = '-'.join(labels)
    filename = os.path.join(directory, 'PCAMatch' + name + '.eps')
    xsp.datadefs.image.saveAllAsLineImages(filename, selectImages)

def getExptPCAData(means):
    exptImages = getAllExptImages()
    labels = [im.label for im in exptImages]
    freqMatMean = np.array([im.frequencies for im in exptImages])
    freqMat = freqMatMean - means
    return (labels, freqMat)

def getPCAData(calcImages):
    labels = [im.label for im in calcImages]
    # data as rows
    freqMatMean = np.array([im.frequencies for im in calcImages])

    means = np.mean(freqMatMean, axis=0)
    freqMat = freqMatMean - means

    (U, s, V) = np.linalg.svd(freqMat)

    return (labels, 
            calcImages[0].distances, 
            means, 
            freqMat, 
            np.power(s, 2), 
            np.transpose(V))

def getAllCalcImages():
    directory = '~/work/all_rfdata_smoothed_unique/'
    filedir = os.path.expanduser(directory)
    return getAllImages(filedir, ['Calc', 'calc'])

def getAllExptImages():
    directory = '~/work/all_rfdata_smoothed_unique/'
    filedir = os.path.expanduser(directory)
    return getAllImages(filedir, ['Expt', 'expt'])

def getAllImages(filedir, filterStrs):
    calcFiles = [os.path.join(filedir, f) for f in os.listdir(filedir)
                   if os.path.isfile(os.path.join(filedir, f))
                   if any((s in f for s in filterStrs))]
    calcImages = [xsp.datadefs.image.fromFile(f) for f in calcFiles]
    return calcImages

def getImage():
    filename = '~/work/all_rfdata_smoothed_unique/SiLiExpt1.dat'
    filename = os.path.expanduser(filename)
    image = xsp.datadefs.image.fromFile(filename)
    print(str(image))

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
# getSynthExptRecognitionAnalysis(int(sys.argv[1]))
plotBestMatches()
