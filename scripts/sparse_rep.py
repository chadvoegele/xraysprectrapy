import os
import sys
import random
import xrayspectrapy as xsp
import numpy as np
import spams as sp
import script_utils as su
import matplotlib.pyplot as plt

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
    calcImages = getAllCalcImages()
    calcImages = xsp.pdf.smooth_images(calcImages, 0.0092)
    calcImages = [xsp.pdf.normalize_image(im) for im in calcImages]
    allImages = [im for sublist in [getAllExptImages(), calcImages] 
                    for im in sublist]

    results = runExptAnalysis()
    results = np.array(results)
    results = results[results[:,0].argsort(),:]

    plotPrefix = 'SparseRep'

    outDir = os.path.expanduser('~/code/xrayspectrapy/doc/autotex')
    filename = 'sparseRepMatchPlots.tex'
    outputFile = os.path.join(outDir, filename)
    f = open(outputFile,'w')

    for result in results:
        imFilename = plotPrefix + '-'.join(result) + '.eps'
        imCaption = ', '.join(result)

        selectImages = [im for im in allImages
                        if any([im.label in l for l in result])]
        outdir = os.path.expanduser('~/code/xrayspectrapy/doc/figs')
        filename = os.path.join(outdir, imFilename)
        xsp.datadefs.image.saveAllAsLineImages(filename, selectImages)

        f.write('\\begin{figure}[ht]\n')
        f.write('\t\\begin{center}\n')
        f.write('\t\t\\includegraphics[scale=0.8]{figs/' + imFilename + '}\n')
        f.write('\t\\caption{Sparse Representation Matches: ' + imCaption + '}\n')
        f.write('\t\\end{center}\n')
        f.write('\\end{figure}\n')

    f.close()

def getSynthExptRecognitionAnalysis(epsilon):
    nSamples = 500
    tSmooth = 0.0092
    direc = os.path.expanduser('~/work/all_rfdata_unique/')
    unSmoothImages = su.getAllImages(direc, ['Calc', 'calc'])
    # unSmoothImages = unSmoothImages[0:100]
    images = xsp.pdf.smooth_images(unSmoothImages, tSmooth)
    images = [xsp.pdf.normalize_image(im) for im in images]
    calcAsCols = np.transpose(np.array([im.frequencies for im in images]))

    mean = 0.004
    stdevs = [0, 0.003, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    # stdevs = [0.01]
    for stdev in stdevs:
        randomNoiseFn = lambda x: xsp.pdf.rand_normal_peaks(x, mean, stdev, 7, 14)
        accuracy = getSynthExptAccuracy(nSamples, unSmoothImages, calcAsCols,
                epsilon, randomNoiseFn, tSmooth)
        print(str(epsilon) + "," + str(mean) + "," + str(stdev) + "," + str(accuracy))

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

    output = []
    for im in exptImages:
        index = findIndexOfMatch(calcCols, im.frequencies, 0.000001)
        calcMatch = calcLabels[index]
        index2 = findIndexOfMatch(calcCols, im.frequencies, 0.000001, 2)
        calcMatch2 = calcLabels[index2]
        output.append([im.label, calcMatch, calcMatch2])

    return output

def exportExptMatchesTable():
    results = runExptAnalysis()
    results = np.array(results)
    results = results[results[:,0].argsort(),:]
    tableRows = [' & '.join(dataRow) + '\\\\ \hline \n' 
                    for dataRow in results]
    outDir = os.path.expanduser('~/code/xrayspectrapy/doc/autotex')
    filename = 'exptSparseRepRecognition.tex'
    outputFile = os.path.join(outDir, filename)
    f = open(outputFile,'w')
    [f.write(row) for row in tableRows]
    f.close()

def plotCompositeExptAnalysis():
    (calcLabels, calcCols) = getCalcDataAsCols()
    exptImages = getAllExptImages()
    exptImages.sort(key=lambda im: im.label)
    plotPrefix = 'SparseRep'

    outDir = os.path.expanduser('~/code/xrayspectrapy/doc/autotex')
    filename = 'sparseRepCompositePlots.tex'
    outputFile = os.path.join(outDir, filename)
    f = open(outputFile,'w')

    for exptImage in exptImages:
        (x, residuals) = findSparseRep(calcCols, exptImage.frequencies, 1e-6)
        compositeMatchFreqs = np.dot(calcCols, x)
        compositeImage = xsp.Image(exptImage.distances, compositeMatchFreqs, 
                'CompositeImage')
        imFilename = plotPrefix + exptImage.label + 'Composite.eps'
        imCaption = exptImage.label

        outdir = os.path.expanduser('~/code/xrayspectrapy/doc/figs')
        filename = os.path.join(outdir, imFilename)
        xsp.datadefs.image.saveAllAsLineImages(filename, 
                [exptImage, compositeImage])

        f.write('\\begin{figure}[ht]\n')
        f.write('\t\\begin{center}\n')
        f.write('\t\t\\includegraphics[scale=0.8]{figs/' + imFilename + '}\n')
        f.write('\t\\caption{Sparse Representation Composite Match: ' + imCaption + '}\n')
        f.write('\t\\end{center}\n')
        f.write('\\end{figure}\n')

    f.close()

def plotSynthAccuracy():
    # mean stdev l1acc l2acc sparseRep01 sparseRep001 SR0001 SR000001
    data = [[0.004, 0,     0.824, 0.872, 0.012,  0.099,  0.421,  0.617], 
            [0.004, 0.003, 0.76,  0.766, 0.023,  0.124,  0.38 ,  0.544],  
            [0.004, 0.01,  0.514, 0.422, 0.018,  0.106,  0.194,  0.295],  
            [0.004, 0.02,  0.276, 0.18,  0.007,  0.093,  0.132,  0.142],  
            [0.004, 0.03,  0.156, 0.122, 0.013,  0.087,  0.093,  0.084],  
            [0.004, 0.04,  0.094, 0.064, 0.009,  0.046,  0.076,  0.07 ],  
            [0.004, 0.05,  0.06,  0.038, 0.006,  0.043,  0.04 ,  0.046],  
            [0.004, 0.06,  0.036, 0.028, 0.01 ,  0.032,  0.037,  0.03 ]]

    data = np.array(data)
    
    plt.plot(data[:,1]/0.004, data[:,2], '-', label='L1 Norm')
    plt.plot(data[:,1]/0.004, data[:,3], '-', label='L2 Norm')
    plt.plot(data[:,1]/0.004, data[:,4], '-', label='e = 0.01')
    plt.plot(data[:,1]/0.004, data[:,5], '-', label='e = 0.001')
    plt.plot(data[:,1]/0.004, data[:,6], '-', label='e = 0.0001')
    plt.plot(data[:,1]/0.004, data[:,7], '-', label='e = 0.000001')

    plt.ylabel('Accuracy')
    plt.xlabel('k x Standard Deviations')
    plt.legend()
    directory = os.path.expanduser('~/code/xrayspectrapy/doc/figs')
    filename = os.path.join(directory, 'SparseRepAccuracyVsLpNorms.eps')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def getCalcDataAsCols():
    images = getAllCalcImages()
    images = xsp.pdf.smooth_images(images, 0.0092)
    images = [xsp.pdf.normalize_image(im) for im in images]
    labels = [im.label for im in images]
    # data as cols
    freqMat = np.transpose(np.array([im.frequencies for im in images]))

    return (labels, freqMat)

def getAllCalcImages():
    directory = '~/work/all_rfdata_unique/'
    filedir = os.path.expanduser(directory)
    return su.getAllImages(filedir, ['Calc', 'calc'])

def getAllExptImages():
    directory = '~/work/all_rfdata_unique/'
    filedir = os.path.expanduser(directory)
    return su.getAllImages(filedir, ['Expt', 'expt'])

def findIndexOfMatch(A, y, epsilon, nthMatch=1):
    (_, residuals) = findSparseRep(A, y, epsilon)
    i = np.argsort(residuals)[nthMatch-1]
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
            'numThreads' : 4,
            'pos' : True,
            'mode' : 1}
    x = sp.lasso(yCol, D=A, **param).toarray()
    residuals = [np.linalg.norm(A[:,i]*x[i] - y) for i in range(0,np.shape(A)[1])]
    return (x, residuals)

# runExptAnalysis()
# getSynthExptRecognitionAnalysis(float(sys.argv[1]))
plotCompositeExptAnalysis()
