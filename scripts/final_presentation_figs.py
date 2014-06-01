import xrayspectrapy as xsp
import numpy as np
import os
import script_utils as su
import matplotlib as mpl
import matplotlib.pyplot as plt

def plotNoisifiedImage():
    tSmooth = 0.0092
    calcIm = getAllCalcImages()[1503]
    calcIm = smoothAndNormalize([calcIm], tSmooth)[0]
    randomNoiseFn = lambda x: xsp.pdf.rand_normal_peaks(x, 0.004, 0.03, 7, 14)
    nSamples = 10

    plt.plot(calcIm.distances, calcIm.frequencies, '-', linewidth=5)
    for i in range(0, nSamples):
        exprIm = xsp.pdf.noisify_image(calcIm, randomNoiseFn, tSmooth)
        plt.plot(exprIm.distances, exprIm.frequencies, '-')

    outdir = os.path.expanduser('~/work/final_figs')
    outfile = os.path.join(outdir, 'RandomImgs.png')
    plt.xlabel('Distances (Angstroms)')
    plt.ylabel('Frequencies')
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()

def plotSiLiMatch():
    calcIms = getAllCalcImages()
    calcIms = smoothAndNormalize(calcIms, 0.0092)
    exptIms = getAllExptImages()
    ims = [im for x in [calcIms, exptIms] for im in x]
    ims = [im for im in ims 
             if any(s  in im.label for s in ['SiLiExpt1', 'SiLiCalc10001'])]
    outdir = os.path.expanduser('~/work/final_figs')
    outfile = os.path.join(outdir, 'SiLiMatch.png')
    xsp.datadefs.image.saveAllAsLineImages(outfile, ims, legendLoc=8)

def plotExptImage():
    ims = getAllExptImages()
    selectIm = [im for im in ims if im.label == 'SiLiExpt6'][0]
    outdir = os.path.expanduser('~/work/final_figs')
    outfile = os.path.join(outdir, 'SelectExptImage.png')
    plt.plot(selectIm.distances, selectIm.frequencies, '-')
    plt.xlabel('Distances (Angstroms)')
    plt.ylabel('Frequencies')
    plt.axis([1, 7, 0, 0.12])
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()

def plotAllCalcImages():
    ims = getAllCalcImages()
    ims = smoothAndNormalize(ims, 0.0092)
    outdir = os.path.expanduser('~/work/final_figs')
    outfile = os.path.join(outdir, 'AllCalcImages.png')
    xsp.datadefs.image.saveAllAsLineImages(outfile, ims, False)

def smoothAndNormalize(ims, t):
    outIms = xsp.pdf.smooth_images(ims, t)
    outIms = [xsp.pdf.normalize_image(im) for im in outIms]
    return outIms

def getAllCalcImages():
    directory = '~/work/all_rfdata_unique/'
    filedir = os.path.expanduser(directory)
    return su.getAllImages(filedir, ['Calc', 'calc'])

def getAllExptImages():
    directory = '~/work/all_rfdata_unique/'
    filedir = os.path.expanduser(directory)
    return su.getAllImages(filedir, ['Expt', 'expt'])

mpl.rcParams['font.size'] = 20
# plotAllCalcImages()
# plotExptImage()
# plotSiLiMatch()
plotNoisifiedImage()
