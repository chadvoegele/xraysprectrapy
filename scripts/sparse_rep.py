import os
import sys
import random
import xrayspectrapy as xsp
import numpy as np
import spams as sp

def test_spams_test_data_A():
    A = [[0.2787,0.4058,0.2954,0.8957,0.9326,0.6305,0.6444,0.1163,0.8589],
         [0.5247,0.7823,0.7004,0.2342,0.1025,0.4604,0.0890,0.8082,0.1115],
         [0.0922,0.9426,0.1918,0.5982,0.9640,0.9684,0.1079,0.4239,0.7053],
         [0.1836,0.4342,0.4407,0.7675,0.8885,0.5800,0.7976,0.7394,0.8873],
         [0.7788,0.0899,0.5127,0.2805,0.0935,0.1858,0.3352,0.4469,0.8943],
         [0.7410,0.2681,0.9841,0.3277,0.0000,0.5392,0.2081,0.9834,0.7845],
         [0.3947,0.6138,0.7817,0.2595,0.9988,0.5365,0.4180,0.4567,0.0339],
         [0.9696,0.9844,0.3625,0.6762,0.0867,0.1756,0.1033,0.7029,0.6719],
         [0.5264,0.5030,0.8024,0.2338,0.4212,0.6630,0.3780,0.5237,0.2908],
         [0.8703,0.0552,0.2679,0.9945,0.1997,0.2248,0.8547,0.6583,0.8348],
         [0.6405,0.8145,0.0379,0.9858,0.4412,0.5277,0.1570,0.3133,0.1101],
         [0.2137,0.5079,0.6625,0.9870,0.5407,0.7011,0.8036,0.9577,0.6859],
         [0.9049,0.2582,0.2980,0.0278,0.2816,0.6093,0.0953,0.3350,0.4152],
         [0.3626,0.8266,0.9937,0.8167,0.2432,0.3784,0.8887,0.8240,0.6648],
         [0.4736,0.4507,0.9633,0.3603,0.7224,0.0689,0.9810,0.2433,0.5713],
         [0.4631,0.9780,0.8029,0.3889,0.2052,0.2919,0.2992,0.9178,0.0214]]
    return A

def test_spams1():
    A = test_spams_test_data_A()
    y = [0.0961, 0.6540, 0.5682, 0.8613, 0.5369, 1.4514, 0.3291, 0.4568, 0.4432,
            0.7847, 0.4048, 1.2412, 0.2894, 0.5179, 0.2680, 0.6873]
    trueI = 7
    i = findIndexOfMatch(A, y, 0.75)
    if (i == trueI):
        print("success")
    else:
        print("fail")

def test_spams2():
    A = test_spams_test_data_A()
    y = [ 0.5211, 0.3726, 1.2979, 0.6756, 0.2232, 0.7958, 0.3867, 0.1141,
            0.5611, 0.2680, 0.6818, 0.9086, 0.5263, 0.2378, 0.0759, 0.2186]
    trueI = 5

    i = findIndexOfMatch(A, y, 0.75)
    if (i == trueI):
        print("success")
    else:
        print("fail")

def test_spams3():
    A = test_spams_test_data_A()
    y = [ 0.7641, 0.5141, 0.5900, 0.3831, 0.1196, 0.2477, 0.6335, 0.5140,
            0.6193, 0.0840, 0.8627, 0.8444, 0.4800, 1.2122, 0.4878, 0.9723 ]
    trueI = 1

    i = findIndexOfMatch(A, y, 0.75)
    if (i == trueI):
        print("success")
    else:
        print("fail")

def plotBestMatches():
    images = [getAllCalcImages(), getAllExptImages()]
    images = [im for sublist in images for im in sublist]

    bestMatches = [['SiLiExpt3', 'SiLiCalc10001'],
                   ['SiLiExpt5', 'SiLiCalc10003'],
                   ['SiLiExpt6', 'SiLiCalc10616'],
                   ['ExptInAs', 'CalcInAs'],
                   ['SiLiExpt7', 'SiLiCalc10382'],
                   ['SiLiExpt8', 'SiLiCalc10382'],
                   ['ExptGaAs', 'CalcGaAs'],
                   ['SiLiExpt2', 'SiLiCalc10001'],
                   ['SiLiExpt4', 'SiLiCalc10003'],
                   ['SiLiExpt1', 'SiLiCalc10001']]
    
    for labels in bestMatches:
        plotImages(labels, images, 'SparseRep')

def getSynthExptRecognitionAnalysis(epsilon):
    nSamples = 500
    tSmooth = 0.004
    direc = os.path.expanduser('~/work/all_rfdata_unique/')
    unSmoothImages = getAllImages(direc, ['Calc', 'calc'])
    # unSmoothImages = unSmoothImages[0:100]
    images = [xsp.pdf.smooth_image(im, tSmooth) for im in unSmoothImages]
    images = [xsp.pdf.normalize_image(im) for im in images]
    calcAsCols = np.transpose(np.array([im.frequencies for im in images]))

    stdevs = [0, 0.003, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    for stdev in stdevs:
        randomNoiseFn = lambda x: xsp.pdf.rand_normal_peaks(x, 0, stdev, 7, 14)
        accuracy = getSynthExptAccuracy(nSamples, unSmoothImages, calcAsCols,
                epsilon, randomNoiseFn, tSmooth)
        print(str(stdev) + "," + str(accuracy))

def getSynthExptAccuracy(nSamples, unSmoothImages, calcAsCols,  
        epsilon, randomNoiseFn, tSmooth):
    success = 0
    for i in range(0, nSamples):
        idx = random.randrange(0, len(unSmoothImages))
        randImg = unSmoothImages[idx]
        exprImg = xsp.pdf.noisify_image(randImg, randomNoiseFn, tSmooth)
        matchIdx = findIndexOfMatch(calcAsCols, exprImg.frequencies, epsilon)
        if idx == matchIdx:
            success = success + 1

    return success / nSamples

def runExptAnalysis():
    (calcLabels, calcCols) = getCalcDataAsCols()
    exptImages = getAllExptImages()

    for im in exptImages:
        index = findIndexOfMatch(calcCols, im.frequencies, 0.000001)
        calcMatch = calcLabels[index]
        print("Expt: " + im.label + "; Match: " + calcMatch)

def getCalcDataAsCols():
    images = getAllCalcImages()
    labels = [im.label for im in images]
    # data as cols
    freqMat = np.transpose(np.array([im.frequencies for im in images]))

    return (labels, freqMat)

def getAllCalcImages():
    directory = '~/work/all_rfdata_smoothed_unique/'
    filedir = os.path.expanduser(directory)
    return getAllImages(filedir, ['Calc', 'calc'])

def getAllExptImages():
    directory = '~/work/all_rfdata_smoothed_unique/'
    filedir = os.path.expanduser(directory)
    return getAllImages(filedir, ['Expt', 'expt'])

def findIndexOfMatch(A, y, epsilon):
    (_, residuals) = findSparseRep(A, y, epsilon)
    i = np.argmin(residuals)
    return i

def findIndexOfMatchCoef(A, y, epsilon):
    (x, residuals) = findSparseRep(A, y, epsilon)
    i = np.argmax(x)
    return i

def findSparseRep(A, y, epsilon):
    # A: data in columns
    # y: image to match
    # epsilon: allowed error
    yCol = np.asfortranarray(np.reshape(y, (len(y), 1)), dtype=float)
    A = np.asfortranarray(np.array(A), dtype=float)
    param = {'lambda1' : epsilon,
            'numThreads' : -1,
            'pos' : True,
            'mode' : 1}
    x = sp.lasso(yCol, D=A, **param).toarray()
    residuals = [np.linalg.norm(A[:,i]*x[i] - y) for i in range(0,np.shape(A)[1])]
    return (x, residuals)

# runExptAnalysis()
plotBestMatches()
# getSynthExptRecognitionAnalysis(float(sys.argv[1]))
# plotBestMatches()
# ims = getAllExptImages()
# print(str([im for im in ims if im.label == 'ExptGaAs'][0]))
