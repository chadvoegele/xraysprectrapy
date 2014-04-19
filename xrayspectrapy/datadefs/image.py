"""Holds x-ray scattering image data"""
class Image:
    def __init__(self, distances, frequencies):
        if (len(distances) != len(frequencies)):
            raise ValueError('distances and frequencies must have same length')
        self.distances = distances
        self.frequencies = frequencies

    def __str__(self):
        out_str = 'Image: '
        for pair in zip(self.distances, self.frequencies):
            out_str = out_str + '\n'
            out_str = out_str + str(pair[0]) + '\t' + str(pair[1])
        return out_str

