import sys, os
parentdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parentdir)

import xrayspectrapy as xsp
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

    #for afile in exptFiles:
    #    filepath = os.path.expanduser(os.path.join(directory, afile))
    #    (directory, filename) = os.path.split(filepath)
    #    (filename, _) = os.path.splitext(filename)
    #    imgFilePath = os.path.join(directory, filename + '.jpg')
    #    im = xsp.datadefs.image.fromFile(filepath)
    #    xsp.datadefs.image.saveAsSpectrumImage(imgFilePath, im, 50, 20)

    exptImages = [xsp.datadefs.image.fromFile(f) for f in exptFiles]

    img = exptImages[3]
    # print(str(img))
    # (maxpeak, minpeak) = peakdet(img.frequencies, 0.0001, img.distances)
    # print('\n'.join([str(x) + '\t' + str(y) for (x,y) in maxpeak]))

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

    matches = [('ExptGaAs.dat', 'CalcGaAs.dat'),
               ('ExptInAs.dat', 'CalcInAs.dat'),
               ('SiLiExpt1.dat', 'SiLiCalc10001.dat')]

    allPeakDiffs = []
    for match in matches:
        exptimg = xsp.datadefs.image.fromFile(os.path.join(filedir, match[0]))
        calcimg = xsp.datadefs.image.fromFile(os.path.join(filedir, match[1]))
        diffFreq = [e - c 
                for (e, c) in zip(exptimg.frequencies, calcimg.frequencies)]
        diffImg = xsp.Image(exptimg.distances, diffFreq)
        (maxpeak, minpeak) = peakdet(diffImg.frequencies, 0.0001, 
                diffImg.distances)
        peakDiffs = [p[1] for p in maxpeak]
        allPeakDiffs.extend(peakDiffs)

    print('\n'.join((str(p) for p in allPeakDiffs)))
    # hist = np.histogram(allPeakDiffs, [-0.01 + 0.001*d for d in range(0,22)])
    # print('\n'.join([str(b[1]) + '\t' + str(b[0]) 
        # for b in zip(hist[0], hist[1])]))
 
# https://gist.github.com/endolith/250860
def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = arange(len(v))
    
    v = asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True
 
    return array(maxtab), array(mintab)

main()
