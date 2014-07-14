import xrayspectrapy as xsp
import os
import sys
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import script_utils as su

def main():
    (_, _, mean, freqMat, eigenvalues, V) = getPCAData(getAllCalcImages())

    (_, freqExpt) = getExptPCAData(mean)
    TExp = np.dot(freqExpt, V)

    # freqMatFromPCs = np.dot(np.dot(freqMat, V), np.transpose(V)) + mean
    print(matToStr(TExp[:,0:2]))

def getExptRecognitionAccuracy(nPCs = 128):
    (labels, _, mean, freqMat, _, V) = getPCAData(getAllCalcImages())
    T = np.dot(freqMat, V)[:,0:nPCs]

    (exptLabels, freqExpt) = getExptPCAData(mean)
    TExp = np.dot(freqExpt, V)[:,0:nPCs]
    
    output = [[x for x in exptLabels]]
    for i in range(0,5):
        findMatch = lambda x: getIndexOfNthClosestVec(T, x, i+1)
        exptMatches = [labels[findMatch(x)] for x in TExp]
        output.append(exptMatches)

    output = np.transpose(np.array(output))
    output = output[output[:,0].argsort(),:]
    tableRows = [' & '.join(dataRow) + '\\\\ \hline \n' 
                    for dataRow in output]
    outDir = os.path.expanduser('~/code/xrayspectrapy/doc/autotex')
    filename = 'expt_recog_with_' + str(nPCs) + 'pcs.tex'
    outputFile = os.path.join(outDir, filename)
    f = open(outputFile,'w')
    [f.write(row) for row in tableRows]
    f.close()

def getSynthExptRecognitionAnalysis(nPCs):
    nSamples = 500
    tSmooth = 0.0092
    direc = os.path.expanduser('~/work/all_rfdata_unique/')
    unSmoothImages = su.getAllImages(direc, ['Calc', 'calc'])
    images = xsp.pdf.smooth_images(unSmoothImages, tSmooth)
    images = [xsp.pdf.normalize_image(im) for im in images]
    (labels, _, mean, freqMat, _, V) = getPCAData(images)
    T = np.dot(freqMat, V)[:,0:nPCs]

    noiseMean = 0.004
    stdevs = [0, 0.003, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    # stdevs = [0.03]
    for stdev in stdevs:
        randomNoiseFn = lambda x: xsp.pdf.rand_normal_peaks(x, noiseMean, stdev, 7, 14)
        accuracy = getSynthExptAccuracy(nSamples, nPCs, unSmoothImages, T, V,
                mean, randomNoiseFn, tSmooth)
        print(str(noiseMean) + "," + str(stdev) + "," + str(accuracy))

def getSynthExptAccuracy(nSamples, nPCs, unSmoothImages, T, V,
        mean, randomNoiseFn, tSmooth):
    success = 0
    for i in range(0, nSamples):
        idx = random.randrange(0, len(unSmoothImages))
        randImg = unSmoothImages[idx]
        exprImg = xsp.pdf.noisify_image(randImg, randomNoiseFn, tSmooth)
        TExpr = np.dot(exprImg.frequencies - mean, V)[0:nPCs]
        matchIdx = getIndexOfNthClosestVec(T, TExpr, 1)
        if idx == matchIdx:
            success = success + 1

    return success / nSamples

def getIndexOfNthClosestVec(vecs, otherVec, n):
    # 1st closest: 0 index
    # 2nd closest: 1 index
    return np.argsort([np.linalg.norm(vec - otherVec, ord=2) for vec in vecs])[n-1]

def printAllCalcFreqs():
    calcImages = getAllCalcImages()
    freqMatMean = np.array([im.frequencies for im in calcImages])
    print(matToStr(freqMatMean))

def exportExptEigenspaceTables():
    (calcLabels, _, mean, freqMat, eigenvalues, V) = getPCAData(getAllCalcImages())
    (exptLabels, freqExpt) = getExptPCAData(mean)
    TExp = np.dot(freqExpt, V)

    for i in range(0, 2):
        sortIdx = TExp[:,i].argsort()
        dataAsStr = [['%.4f' % x for x in row] for row in TExp[sortIdx,0:2]]
        tableRows = [label + ' & ' + ' & '.join(data) + '\\\\ \hline \n' 
                        for (label, data) 
                        in zip(np.array(exptLabels)[sortIdx], dataAsStr)]

        outDir = os.path.expanduser('~/code/xrayspectrapy/doc/autotex')
        filename = 'expt_eigen_sort_by_pc' + str(i+1) + '.tex'
        outputFile = os.path.join(outDir, filename)
        f = open(outputFile,'w')
        [f.write(row) for row in tableRows]
        f.close()

def plotEigenSpaceScatter():
    (calcLabels, _, mean, freqMat, eigenvalues, V) = getPCAData(getAllCalcImages())
    T = np.dot(freqMat, V)

    (exptLabels, freqExpt) = getExptPCAData(mean)
    TExp = np.dot(freqExpt, V)

    pairs = ((0,1,4),
             (1,2,2),
             (2,3,2),
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

def plotEigenface(eigenIndex):
    (_, dists, means, freqMat, eigenvalues, V) = getPCAData(getAllCalcImages())

    eigenface = V[:,eigenIndex - 1]

    allLocs = [3, 1, 4]

    n = 5
    maxLoad = 0.15
    minLoad = -maxLoad
    step = (maxLoad - minLoad) / (n - 1)

    load = minLoad
    legend = []
    for i in range(0,n):
        plt.plot(dists, eigenface * load + means, '-')
        legend.append('Loading: ' + str(load))
        load = load + step

    plt.ylabel('Frequencies')
    plt.xlabel('Distances (Angstroms)')
    plt.legend(legend, loc=allLocs[eigenIndex - 1])
    directory = os.path.expanduser('~/code/xrayspectrapy/doc/figs')
    filename = os.path.join(directory, 'eigenface' + str(eigenIndex) + '.eps')
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
    direc = os.path.expanduser('~/code/xrayspectrapy/doc/figs')
    filename = os.path.join(direc, 'all_calc_images_mean.png')
    plt.savefig(filename, dpi=250, bbox_inches='tight')
    plt.close()

def plotBestMatches(nPCs = 128):
    calcImages = getAllCalcImages()
    allImages = [calcImages, getAllExptImages()]
    allImages = [im for sublist in allImages for im in sublist]

    (labels, _, mean, freqMat, _, V) = getPCAData(calcImages)
    T = np.dot(freqMat, V)[:,0:nPCs]
    (exptLabels, freqExpt) = getExptPCAData(mean)
    TExp = np.dot(freqExpt, V)[:,0:nPCs]

    exptLabels = np.array(exptLabels)
    sortIdx = exptLabels.argsort()
    exptLabels = exptLabels[sortIdx]
    TExp = TExp[sortIdx,:]
    
    outDir = os.path.expanduser('~/code/xrayspectrapy/doc/autotex')
    filename = 'eigen_match_plots_' + str(nPCs) + 'PCs.tex'
    outputFile = os.path.join(outDir, filename)
    f = open(outputFile,'w')

    for i in range(0,len(exptLabels)):
        label = exptLabels[i]
        oneTExp = TExp[i,:]
        findBestMatch = lambda x: getIndexOfNthClosestVec(T, x, 1)
        bestMatchLabel = labels[findBestMatch(oneTExp)]

        findSecondBestMatch = lambda x: getIndexOfNthClosestVec(T, x, 2)
        secondBestMatchLabel = labels[findSecondBestMatch(oneTExp)]

        matchLabels = [label, bestMatchLabel, secondBestMatchLabel]
        plotPrefix = 'PC' + str(nPCs) + 'Match'
        plotImages(matchLabels, allImages, plotPrefix)

        imFilename = plotPrefix + '-'.join(matchLabels)
        imCaption = ', '.join(matchLabels)

        f.write('\\begin{figure}[ht]\n')
        f.write('\t\\begin{center}\n')
        f.write('\t\t\\includegraphics[scale=0.8]{figs/' + imFilename + '.eps}\n')
        f.write('\t\\caption{PCA Matches: ' + imCaption + '}\n')
        f.write('\t\\end{center}\n')
        f.write('\\end{figure}\n')

    f.close()

def plotOutliers(eigenIdx):
    (labels, dists, mean, freqMat, _, V) = getPCAData(getAllCalcImages())

    outLoad = 0.05
    T = np.dot(freqMat, V)

    Tneg = [t for t in T if t[eigenIdx-1] <= -outLoad]
    Tmid = [t for t in T if t[eigenIdx-1] > -outLoad and t[eigenIdx-1] < outLoad]
    Tpos = [t for t in T if t[eigenIdx-1] >= outLoad]

    legend = []
    if np.shape(Tneg)[0] > 0:
        freqMatFromPCsNeg = np.dot(Tneg, np.transpose(V)) + mean
        meanNeg = np.mean(freqMatFromPCsNeg, axis=0)
        plt.plot(dists, meanNeg, '-')
        legend.append('$PC(' + str(eigenIdx) + ') \leq ' + str(-outLoad) + '$')

    if np.shape(Tmid)[0] > 0:
        freqMatFromPCsMid = np.dot(Tmid, np.transpose(V)) + mean
        meanMid = np.mean(freqMatFromPCsMid, axis=0)
        plt.plot(dists, meanMid, '-')
        legend.append('$' + str(-outLoad) + ' < PC(' + str(eigenIdx) + ') < ' 
                        + str(outLoad) + '$')

    if np.shape(Tpos)[0] > 0:
        freqMatFromPCsPos = np.dot(Tpos, np.transpose(V)) + mean
        meanPos = np.mean(freqMatFromPCsPos, axis=0)
        plt.plot(dists, meanPos, '-')
        legend.append('$' + str(outLoad) + ' \geq PC(' + str(eigenIdx) + ')$')

    plt.ylabel('Frequencies')
    plt.xlabel('Distances (Angstroms)')
    plt.legend(legend)
    directory = os.path.expanduser('~/code/xrayspectrapy/doc/figs')
    filename = os.path.join(directory, 'eigenOutlier' + str(eigenIdx) + '.eps')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plotSynthAccuracyByPrinComp():
    # nPCs accuracyRun1 accuracyRun2 ...
    data =  [[3,       0.018,   0.020,   0.012,   0.022,   0.016],
             [5,       0.028,   0.034,   0.034,   0.034,   0.028],
             [8,       0.048,   0.044,   0.060,   0.058,   0.062],
             [16,      0.092,   0.104,   0.116,   0.096,   0.084],
             [24,      0.108,   0.142,   0.122,   0.102,   0.120],
             [36,      0.114,   0.112,   0.110,   0.110,   0.112],
             [48,      0.126,   0.114,   0.098,   0.114,   0.098],
             [60,      0.102,   0.138,   0.112,   0.096,   0.096],
             [72,      0.120,   0.086,   0.104,   0.090,   0.106],
             [84,      0.116,   0.096,   0.114,   0.102,   0.130],
             [96,      0.096,   0.096,   0.122,   0.112,   0.126],
             [108,     0.100,   0.114,   0.118,   0.108,   0.102],
             [128,     0.116,   0.104,   0.102,   0.114,   0.096]]
    data = np.array(data)
    
    for i in range(1, np.shape(data)[1]):
        plt.plot(data[:,0], data[:,i], '-')

    plt.ylabel('Accuracy')
    plt.xlabel('Number of Principal Components')
    directory = os.path.expanduser('~/code/xrayspectrapy/doc/figs')
    filename = os.path.join(directory, 'accuracyByNumberOfPCs.eps')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plotSynthAccuracy():
    # mean stdev l1acc l2acc pc128l1accuracy pc128l2accuracy
    data = [[0.004, 0,     0.824, 0.872, 0.896, 0.878],
            [0.004, 0.003, 0.76,  0.766, 0.808, 0.78],
            [0.004, 0.01,  0.514, 0.422, 0.456, 0.432],
            [0.004, 0.02,  0.276, 0.18,  0.212, 0.186],
            [0.004, 0.03,  0.156, 0.122, 0.086, 0.094],
            [0.004, 0.04,  0.094, 0.064, 0.078, 0.096],
            [0.004, 0.05,  0.06,  0.038, 0.042, 0.032],
            [0.004, 0.06,  0.036, 0.028, 0.014, 0.024]]
    data = np.array(data)
    
    plt.plot(data[:,1]/0.004, data[:,2], '-', label='L1 Norm')
    plt.plot(data[:,1]/0.004, data[:,3], '-', label='L2 Norm')
    plt.plot(data[:,1]/0.004, data[:,4], '-', label='PCA - L1')
    plt.plot(data[:,1]/0.004, data[:,5], '-', label='PCA - L2')

    plt.ylabel('Accuracy')
    plt.xlabel('k x Standard Deviations')
    plt.legend()
    directory = os.path.expanduser('~/code/xrayspectrapy/doc/figs')
    filename = os.path.join(directory, 'PCAAccuracyVsLpNorms.eps')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plotImages(labels, images, filePrefix):
    selectImages = [im for im in images if im.label in labels]
    directory = os.path.expanduser('~/code/xrayspectrapy/doc/figs')
    name = '-'.join(labels)
    filename = os.path.join(directory, filePrefix + name + '.eps')
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
    directory = '~/work/all_rfdata_unique/'
    filedir = os.path.expanduser(directory)
    ims = su.getAllImages(filedir, ['Calc', 'calc'])
    ims = xsp.pdf.smooth_images(ims, 0.0092)
    ims = [xsp.pdf.normalize_image(im) for im in ims]
    return ims

def getAllExptImages():
    directory = '~/work/all_rfdata_unique/'
    filedir = os.path.expanduser(directory)
    return su.getAllImages(filedir, ['Expt', 'expt'])

# mpl.rcParams['font.size'] = 20
# getSynthExptRecognitionAnalysis(int(sys.argv[1]))
plotSynthAccuracy()
