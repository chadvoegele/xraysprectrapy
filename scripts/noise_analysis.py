import xrayspectrapy as xsp
import os
import math
import numpy as np
import script_utils as su
import matplotlib.pyplot as plt
import scipy.stats as spstats

import sys

def plot_baseline_example():
    ims = su.getAllImages(os.path.expanduser('~/work/all_rfdata_unique/'))
    calcinas = [im for im in ims if im.label == 'CalcInAs'][0]
    exptinas = [im for im in ims if im.label == 'ExptInAs'][0]
    calcinas = xsp.pdf.smooth_image(calcinas, 0.0092)

    d = 0.0001
    (_, mincalc) = xsp.peakdet.peakdet(calcinas.frequencies, d, calcinas.distances)
    (_, minexpt) = xsp.peakdet.peakdet(exptinas.frequencies, d, exptinas.distances)

    mincalc_dists = [d[0] for d in mincalc]
    mincalc_freqs = [d[1] for d in mincalc]

    minexpt_dists = [d[0] for d in minexpt]
    minexpt_freqs = [d[1] for d in minexpt]

    plt.plot(calcinas.distances, calcinas.frequencies, 'r--');
    plt.plot(mincalc_dists, mincalc_freqs, 'r-', label='Calc Baseline');
    plt.plot(exptinas.distances, exptinas.frequencies, 'g--');
    plt.plot(minexpt_dists, minexpt_freqs, 'g-', label='Expt Baseline');
    plt.legend()

    filename = os.path.expanduser('~/code/xrayspectrapy/doc/figs/inas_valleys.eps')
    plt.xlabel('Distances (Angstroms)')
    plt.ylabel('Frequencies')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def peak_locs_dist():
    im_dir = os.path.expanduser('~/work/all_rfdata_unique/')
    expt_ims = su.getAllImages(im_dir, ['Expt', 'expt'])
    expt_ims = [xsp.pdf.normalize_image(im) for im in expt_ims]
    delta = 0.0001
    distances = [maxpeak[0] for im in expt_ims
        for maxpeak in xsp.peakdet.peakdet(im.frequencies, delta, im.distances)[0]]
    hist = np.histogram(distances)

    filename = '~/code/xrayspectrapy/doc/figs/peak_locations_hist.eps'
    plt.hist(distances, 10, normed=True)
    plt.xlabel('Distances (Angstroms)')
    plt.ylabel('Frequencies')
    plt.savefig(os.path.expanduser(filename), bbox_inches='tight')
    plt.close()

def peak_counts_dist():
    im_dir = os.path.expanduser('~/work/all_rfdata_unique/')
    expt_ims = su.getAllImages(im_dir, ['Expt', 'expt'])
    expt_ims = [xsp.pdf.normalize_image(im) for im in expt_ims]
    delta = 0.0001
    peakcounts = [len(xsp.peakdet.peakdet(im.frequencies, delta, im.distances)[0]) 
           for im in expt_ims]
    filename = '~/code/xrayspectrapy/doc/figs/peak_counts_hist.eps'
    bins = [x for x in range(7,16)]
    plt.hist(peakcounts, bins, normed=True)
    plt.xlabel('Number of Peaks')
    plt.ylabel('Frequencies')
    plt.savefig(os.path.expanduser(filename), bbox_inches='tight')
    plt.close()

def noise_peak_heights_dist():
    matches = [('ExptGaAs.dat', 'CalcGaAs.dat'),
               ('ExptInAs.dat', 'CalcInAs.dat'),
               ('SiLiExpt1.dat', 'SiLiCalc10001.dat')]

    filedir = os.path.expanduser('~/work/all_rfdata_unique/')
    allPeakDiffs = []
    for match in matches:
        exptimg = xsp.datadefs.image.fromFile(os.path.join(filedir, match[0]))
        calcimg = xsp.datadefs.image.fromFile(os.path.join(filedir, match[1]))
        calcimg = xsp.pdf.smooth_image(calcimg, 0.0092)
        calcimg = xsp.pdf.normalize_image(calcimg)
        diffFreq = [e - c 
                for (e, c) in zip(exptimg.frequencies, calcimg.frequencies)]
        diffImg = xsp.Image(exptimg.distances, diffFreq)
        delta = 0.0001
        (maxpeak, minpeak) = xsp.peakdet.peakdet(diffImg.frequencies, delta, 
                diffImg.distances)
        peakDiffs = [p[1] for p in maxpeak]
        allPeakDiffs.extend(peakDiffs)

    filename = '~/code/xrayspectrapy/doc/figs/noise_peak_dist.eps'
    plt.hist(allPeakDiffs, 25, normed=True)
    plt.xlabel('Peak Differences')
    plt.ylabel('Frequencies')
    plt.savefig(os.path.expanduser(filename), bbox_inches='tight')
    plt.close()

    # http://stackoverflow.com/questions/3209362/how-to-plot-empirical-cdf-in-matplotlib-in-python
    (counts, bins) = np.histogram(allPeakDiffs, bins=25)
    counts = [c / sum(counts) for c in counts]
    cdf = np.cumsum(counts)

    mean = np.mean(allPeakDiffs)
    stdev = np.std(allPeakDiffs)
    print("mean: " + str(mean))
    print("stdev: " + str(stdev))
    normcdf = [spstats.norm.cdf(p, loc=mean, scale=stdev) for p in bins[1:]]

    plt.plot(bins[1:], cdf, 'r-', label='Experimental CDF');
    plt.plot(bins[1:], normcdf, 'g-', label='Normal CDF');
    plt.legend(loc=5)

    filename = '~/code/xrayspectrapy/doc/figs/noise_peak_cdf.eps'
    plt.xlabel('Peak Differences')
    plt.ylabel('Cumulative Frequencies')
    plt.savefig(os.path.expanduser(filename), bbox_inches='tight')
    plt.close()

def main():
    exptImages = [xsp.datadefs.image.fromFile(f) for f in exptFiles]

    img = exptImages[0]
    img = xsp.pdf.normalize_image(img)
    print(str(img))
    (maxpeak, minpeak) = peakdet(img.frequencies, 0.0001, img.distances)
    print('\n'.join([str(x[0]) + '\t' + str(x[1]) for x in maxpeak]))

    #print('\n'.join((str(p) for p in allPeakDiffs)))
    # hist = np.histogram(allPeakDiffs, [-0.01 + 0.001*d for d in range(0,22)])
    # print('\n'.join([str(b[1]) + '\t' + str(b[0]) 
        # for b in zip(hist[0], hist[1])]))
 
# main()
# plot_baseline_example()
# peak_counts()
# peak_locs_dist()
# peak_counts_dist()
noise_peak_heights_dist()
