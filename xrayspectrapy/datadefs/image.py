import numpy as np
import os

"""Holds x-ray scattering image data"""
class Image:
    def __init__(self, distances, frequencies, label=''):
        if (len(distances) != len(frequencies)):
            raise ValueError('distances and frequencies must have same length')
        self.distances = distances
        self.frequencies = frequencies
        self.label = label

    def __str__(self):
        if self.label != '':
            out_str = 'Image (' + self.label + '):'
        else:
            out_str = 'Image: '
        for pair in zip(self.distances, self.frequencies):
            out_str = out_str + '\n'
            out_str = out_str + str(pair[0]) + '\t' + str(pair[1])
        return out_str

def fromFile(filepath, labelPrefix = ''):
    fileContents = np.loadtxt(filepath)
    (filename, _) = os.path.splitext(os.path.basename(filepath))
    return Image([x[0] for x in fileContents],
                 [x[1] for x in fileContents],
                 labelPrefix + filename)

def toFile(filename, image):
    data = [(d, f) for (d, f) in zip(image.distances, image.frequencies)]
    np.savetxt(filename, data, fmt='%.2f %.8f')
