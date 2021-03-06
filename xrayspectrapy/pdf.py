import math
import numpy as np
import xrayspectrapy as xsp
import random

"""Calculations pertaining to pair distribution functions.

Manipulations of PDF Images
---------------------------
- `normalize_image` -- normalizes an image to be non-negative and sum to 1
- `smooth_image` -- smoothes an image with a gaussian blur

Atom locations to PDF Image
---------------------------
- `calc_pdf` -- calculates the pdf image from atom locations

"""

def rand_normal_peaks(image, peak_height_mean=0.004, peak_height_stdev=0.004, 
                      peak_count_min=7, peak_count_max=14):
    countPeaks = random.randint(peak_count_min, peak_count_max)
    peaks = []
    for i in range(0, countPeaks):
        peakHeight = random.normalvariate(peak_height_mean, peak_height_stdev)
        peakDistIndx = random.randrange(0, len(image.distances))
        peakLocation = image.distances[peakDistIndx]
        peaks.append([peakLocation, peakHeight])
    return peaks

def noisify_image(image, rand_peak_fn=rand_normal_peaks, smooth=0.004):
    freqsAndNoise = rand_peak_fn(image) 
    originalFreqs = [[d,f] for (d,f) in zip(image.distances, image.frequencies)]
    freqsAndNoise.extend(originalFreqs)

    newFreqs = []
    for dist in image.distances:
        newFreqs.append(sum([d[1] for d in freqsAndNoise if d[0]==dist]))

    newIm = xsp.Image(image.distances, newFreqs)
    newIm = xsp.pdf.smooth_image(newIm, smooth)
    newIm = normalize_image(newIm)

    return newIm

def normalize_image(image):
    """Normalizes an image to be =>0 and sum to 1."""
    minFreq = min(image.frequencies)
    newFreq = [f - minFreq for f in image.frequencies]
    sumFreq = sum(newFreq)
    newFreq = [f / sumFreq for f in newFreq]
    return xsp.Image(image.distances, newFreq, image.label)

def smooth_image(image, t):
    """Smoothes the image using a gaussian blur according to the smoothing
       constant t."""
    return xsp.Image(image.distances, 
             gaussian_blur(image.distances, image.frequencies, t),
             image.label)

def smooth_images(images, t):
    """Smoothes all images using a gaussian blur according to the smoothing
       constant t."""

    for i in range(1, len(images)):
        if (images[0].distances != images[i].distances):
            raise ValueError('distances must be the same for all images')

    def calc_weights(x, xs, t):
        return [math.exp(-(x-a)*(x-a)/4/t) for a in xs]

    xs = images[0].distances
    weights_mat = np.array([calc_weights(a, xs, t) for a in xs])
    weights_mat = np.divide(weights_mat, sum(weights_mat))
    ys_mat = np.array([im.frequencies for im in images])
    smoothed_ys_mat = np.dot(ys_mat, weights_mat)

    output = [xsp.Image(xs, smoothed_ys, im.label)
                for (im, smoothed_ys) in zip(images, smoothed_ys_mat)]
    return output

def gaussian_blur(xs, ys, t):
    """For two vectors, xs and ys, and a smoothing constant,t, gaussian_blur
       calculates the weierstrass transform of the function xs -> ys."""
    if len(xs) != len(ys):
        raise ValueError("x and y must have the same number of elements")

    def calc_weights(x, xs, t):
        return [math.exp(-(x-a)*(x-a)/4/t) for a in xs]

    def calc_smoothed_y(x, xs, ys, t):
        weights = calc_weights(x, xs, t)
        sum_weights = sum(weights)

        return np.dot(weights, ys)/sum_weights

    return [calc_smoothed_y(x, xs, ys, t) for x in xs]

def calc_pdf(structure, maxDist = 0, binsOrnBins = 128):
    """Calculates the pair distribution function for structure.

       Arguments: structure -- xrayspectrapy.structure.Structure
                  maxOffset -- int
                  binsOrnBins -- int or sequence

       Example: calc_pdf(structure, 1, [0 0.2 0.4 0.6 0.8 1])
                calc_pdf(structure, 1, 5)
    """
    if (structure.period_length != None):
        distances = calc_distances_with_repetition(structure, maxDist)
    else:
        distances = calc_distances(structure)

    if isinstance(binsOrnBins, int):
        nBins = binsOrnBins
        bins = calc_bins(round(min(distances), 1), round(max(distances), 1), nBins)
    else:
        bins = binsOrnBins

    (frequencies, _) = np.histogram(distances, bins)
    outDistances = bins[:-1]
    frequencies = [f / d / d for (f, d) in zip(frequencies, outDistances)]
    sumFrequencies = sum(frequencies)
    frequencies = [f / sumFrequencies for f in frequencies]
    return xsp.Image(outDistances, frequencies)

def calc_bins(lowerBnd, upperBnd, nBins):
    """Returns nBins number of intervals equally spaced between lowerBnd and
       upperBnd.

       Example: calc_bins(0.2, 1, 4) -> [.2 .4 .6 .8 1]
       
    """
    increment = (upperBnd - lowerBnd) / nBins
    return [x * increment + lowerBnd for x in range(0, nBins + 1)]

def calc_distances_with_repetition(structure, maxDist, wantLabels = False):
    """Calculates all pairwise distances up to maxDist between atoms in
       structure where the atoms can be displaced according to a periodicity of 1
       in all directions.

       In 2D for nPeriods = 1,
          --------------------------
          |        |        |      |
       (-1,1)----(0,1)----(1,1)-----
          |        |        |      |
       (-1,0)----(0,0)----(1,0)-----
          |        |        |      |
       (-1,-1)---(0,-1)---(1,-1)----
       """

    atoms = structure.atoms
    period = structure.period_length
    def dist(atom, oAtom): 
        return calc_close_distances(atom, oAtom, period, maxDist, wantLabels)
    distances = [x for atom in atoms for oAtom in atoms for x in dist(atom, oAtom)]
    return distances

def calc_close_distances(thisAtom, otherAtom, period, maxDist, wantLabels=False):
    """Calculates distances up to maxDist from thisAtom to otherAtom where
       otherAtom is displaced according to a periodicity of period in all 
       directions."""
    maxPossibleOffset = math.ceil(maxDist/period) + 1
    r = range(-maxPossibleOffset, maxPossibleOffset + 1)
    offsets = [period * np.array((i,j,k)) for i in r for j in r for k in r]

    zeroOffset = np.array((0,0,0))
    distances = (calc_distance(thisAtom, otherAtom, zeroOffset, o, wantLabels)
                    for o in offsets)
    if wantLabels:
        return [d for d in distances if d.distance <= maxDist if d.distance > 0]
    else:
        return [d for d in distances if d <= maxDist if d > 0]

def calc_distances(structure, wantLabels = False):
    """Calculates all pairwise distances between atoms in structure."""
    atoms = structure.atoms
    zero = np.array((0,0,0))
    return [calc_distance(atom, oAtom, zero, zero, wantLabels) 
             for atom in atoms for oAtom in atoms]

def calc_distance(atom1, atom2, offset1, offset2, wantLabels = False):
    """Calculates distance between atom1 and atom2 where atom1 is offset by
        offset1 and atom2 if offset by offset2.
       dist(atom2 + offset2, atom1 + offset1)

       Arguments:
       atom1, 2 -- xrayspectrapy.atom.Atom
       offset1, 2 -- numpy.ndarray
    """
    a1 = np.array((atom1.x, atom1.y, atom1.z))
    a2 = np.array((atom2.x, atom2.y, atom2.z))
    offsetAtomsDifference = a2 + offset2 - (a1 + offset1)
    if wantLabels:
        return xsp.pdf.DistancesWithLabels(
                np.sqrt(np.inner(offsetAtomsDifference, offsetAtomsDifference)),
                str(atom1.label) + " " + str(offset1),
                str(atom2.label) + " " + str(offset2))
    else:
        return np.sqrt(np.inner(offsetAtomsDifference, offsetAtomsDifference))

class DistancesWithLabels:
    def __init__(self, distance, label1, label2):
        self.distance = distance
        self.label1 = label1
        self.label2 = label2

    def __str__(self):
        return str(self.label1) + ' <-> ' + str(self.label2) + ': ' +\
                str(self.distance)
