import xrayspectrapy as xsp
import os
import math
import numpy as np

import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array

def main():
    directory = '~/work/all_rfdata_smoothed_unique/'
    filedir = os.path.expanduser(directory)
    exptFiles = [os.path.join(filedir, f) for f in os.listdir(filedir)
                   if os.path.isfile(os.path.join(filedir, f))
                   if ('Expt' in f or 'expt' in f)]
    exptFiles = [os.path.expanduser('~/work/inas_fig4n.csv')]

    #for afile in exptFiles:
    #    filepath = os.path.expanduser(os.path.join(directory, afile))
    #    (directory, filename) = os.path.split(filepath)
    #    (filename, _) = os.path.splitext(filename)
    #    imgFilePath = os.path.join(directory, filename + '.jpg')
    #    im = xsp.datadefs.image.fromFile(filepath)
    #    xsp.datadefs.image.saveAsSpectrumImage(imgFilePath, im, 50, 20)

    exptImages = [xsp.datadefs.image.fromFile(f) for f in exptFiles]

    img = exptImages[0]
    img = xsp.pdf.normalize_image(img)
    print(str(img))
    (maxpeak, minpeak) = peakdet(img.frequencies, 0.0001, img.distances)
    print('\n'.join([str(x[0]) + '\t' + str(x[1]) for x in maxpeak]))

    #distances = [d[0] for img in exptImages 
    #    for maxpeak in peakdet(img.frequencies, 0.0001, img.distances)
    #    for d in maxpeak]
    #print('\n'.join([str(d) for d in distances]))
    #hist = np.histogram(distances)
    #print('\n'.join([str(b[1]) + '\t' + str(b[0]) 
    #    for b in zip(hist[0], hist[1])]))

    #peak_counts = [len(peakdet(img.frequencies, 0.0001, img.distances)[0]) 
    #        for img in exptImages]
    #print('\n'.join([str(p) for p in peak_counts]))
    #hist = np.histogram(peak_counts, range(7,16))
    #print('\n'.join([str(b[1]) + '\t' + str(b[0]) 
    #    for b in zip(hist[0], hist[1])]))

    #matches = [('ExptGaAs.dat', 'CalcGaAs.dat'),
    #           ('ExptInAs.dat', 'CalcInAs.dat'),
    #           ('SiLiExpt1.dat', 'SiLiCalc10001.dat')]

    #allPeakDiffs = []
    #for match in matches:
    #    exptimg = xsp.datadefs.image.fromFile(os.path.join(filedir, match[0]))
    #    calcimg = xsp.datadefs.image.fromFile(os.path.join(filedir, match[1]))
    #    diffFreq = [e - c 
    #            for (e, c) in zip(exptimg.frequencies, calcimg.frequencies)]
    #    diffImg = xsp.Image(exptimg.distances, diffFreq)
    #    (maxpeak, minpeak) = peakdet(diffImg.frequencies, 0.0001, 
    #            diffImg.distances)
    #    peakDiffs = [p[1] for p in maxpeak]
    #    allPeakDiffs.extend(peakDiffs)

    #print('\n'.join((str(p) for p in allPeakDiffs)))
    # hist = np.histogram(allPeakDiffs, [-0.01 + 0.001*d for d in range(0,22)])
    # print('\n'.join([str(b[1]) + '\t' + str(b[0]) 
        # for b in zip(hist[0], hist[1])]))
 
main()
