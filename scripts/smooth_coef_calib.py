import script_utils as su
import xrayspectrapy as xsp
import numpy as np
import os
import matplotlib.pyplot as plt

def plotSmoothedImageVsExpt():
    t = 0.0092
    (calcIm, exptIm) = getImages()
    smoothedIm = xsp.pdf.smooth_image(calcIm, t)
    smoothedIm = xsp.pdf.normalize_image(smoothedIm)
    outdir = '~/code/xrayspectrapy/doc/figs/'
    su.plotImages(outdir, [smoothedIm, exptIm], 'SmoothCalibration', 'eps')

def plotErrorCurve():
    data = runSmoothingCoefs(0.001, 0.03, 0.0001)

    ts = [datum[0] for datum in data]
    errors = [datum[1] for datum in data]
    
    plt.plot(ts, errors, '-')
    plt.ylabel('Error')
    plt.xlabel('Smoothing Coefficient')
    directory = os.path.expanduser('~/code/xrayspectrapy/doc/figs')
    filename = os.path.join(directory, 'SmoothingCalibrationCurve.eps')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def printSmoothingCoefs(lower, upper, step):
    outputs = runSmoothingCoefs(lower, upper, step)
    print('t, error')
    for output in outputs:
        print('%.5f' % output[0] + ', ' + '%.5f' % output[1])

def runSmoothingCoefs(lower, upper, step):
    n = round((upper - lower) / step)
    smooth_ts = [lower + i * step for i in range(0,n)]

    (calcIm, exptIm) = getImages()

    outputs = []
    for smooth_t in smooth_ts:
        error = smoothingError(calcIm, exptIm, smooth_t)
        outputs.append((smooth_t, error))
    
    return outputs

def smoothingError(calcIm, exptIm, smooth_coef):
    smoothedIm = xsp.pdf.smooth_image(calcIm, smooth_coef)
    smoothedIm = xsp.pdf.normalize_image(smoothedIm)
    error = xsp.comparers.l2_norm(exptIm, smoothedIm)
    return error

def getImages():
    direc = os.path.expanduser('~/work/all_rfdata_unique')
    calcIm = su.getAllImages(direc, ['SiLiCalc10001'])
    exptIm = su.getAllImages(direc, ['SiLiExpt1'])
    return (calcIm[0], exptIm[0])

# printSmoothingCoefs(0.001, 0.03, 0.0001)
# plotSmoothedImageVsExpt()
plotErrorCurve()
