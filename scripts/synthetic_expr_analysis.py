import sys, os
parentdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parentdir)

import xrayspectrapy as xsp
import math
import numpy as np
import random

import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array

def main():
    directory = '~/work/all_rfdata_smoothed_unique/'
    filedir = os.path.expanduser(directory)
    calcFiles = [os.path.join(filedir, f) for f in os.listdir(filedir)
                   if os.path.isfile(os.path.join(filedir, f))
                   if ('Calc' in f or 'calc' in f)]

    #img = xsp.datadefs.image.fromFile(calcFiles[0])
    #imgFilePath = os.path.join(os.path.expanduser('~/work'), 'rand1.jpg')
    #xsp.datadefs.image.saveAsSpectrumImage(imgFilePath, noisifyImage(img), 50, 20)

    calcImages = [xsp.datadefs.image.fromFile(f) for f in calcFiles]

    labels = [im.label for im in calcImages]
    n = 500
    success = 0
    for i in range(0, n):
        randIndex = random.randrange(0, len(calcImages))
        randImg = calcImages[randIndex]
        exprImg = noisifyImage(randImg)

        dists = [xsp.comparers.l1_norm(exprImg, calcImg)
                 for calcImg in calcImages]
        if np.argmin(dists) == randIndex:
            success = success + 1
    
    print(success / n)


def randPeaks():
    countPeaks = random.randint(7, 14)
    peaks = []
    for i in range(0, countPeaks):
        peakHeight = random.normalvariate(0.0036, 0.0039)
        peakLocation = 1.92 + 0.04*random.randint(0, 127)
        peaks.append([peakLocation, peakHeight])

    return peaks

def normalizeImage(im):
    minFreq = min(im.frequencies)
    newFreq = [f - minFreq for f in im.frequencies]
    sumFreq = sum(newFreq)
    newFreq = [f / sumFreq for f in newFreq]
    return xsp.Image(im.distances, newFreq, im.label)

def noisifyImage(img):
    (maxpeak, minpeak) = peakdet(img.frequencies, 0.0001, img.distances)
    newImPeaks = randPeaks() 
    newImPeaks.extend([[d[0], d[1]] for d in maxpeak])

    newFreqs = []
    for dist in img.distances:
        newFreqs.append(sum([d[1] for d in newImPeaks 
            if (d[0] > dist - 0.001 and d[0] < dist + 0.001)]))

    newIm = xsp.Image(img.distances, newFreqs)
    newIm = normalizeImage(newIm)
    newIm = xsp.pdf.smooth_image(newIm, 0.004)

    return newIm

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
