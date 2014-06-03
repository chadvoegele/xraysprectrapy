import unittest
import xrayspectrapy as xsp

class PeakDetTests(unittest.TestCase):
    def data_for_test1(self):
        dists = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 
                 8.5, 9, 9.5, 10]
        freqs = [0, 0.5, 1, 0.5, 0, 0.1, 0, 10, 9, 8, 7, 0, 0.5, 0, 0.4, 0, 0.8,
                 0.9, 1]
        return xsp.Image(dists, freqs)

    def test_peaks1(self):
        im1 = self.data_for_test1()
        (dists, peaks) = xsp.peakdet.image_peakdet(im1, 0.0001)
        expected_dists = [ 2, 3.5, 4.5, 7, 8 ]
        expected_peaks = [ 1, 0.1, 10, 0.5, 0.4 ]

        for (actual, expected) in zip(dists, expected_dists):
            self.assertAlmostEqual(actual, expected, 4)

        for (actual, expected) in zip(peaks, expected_peaks):
            self.assertAlmostEqual(actual, expected, 4)

    def test_peaks_im1(self):
        im1 = self.data_for_test1()
        peak_im1 = xsp.peakdet.image_as_peaks(im1, 0.0001)
        expected_dists = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7,
                            7.5, 8, 8.5, 9, 9.5, 10]
        expected_freqs = [0, 0, 1, 0, 0, 0.1, 0, 10, 0, 0, 0, 0, 0.5, 0, 0.4, 0,
                          0, 0, 0]

        for (actual, expected) in zip(peak_im1.distances, expected_dists):
            self.assertAlmostEqual(actual, expected, 4)

        for (actual, expected) in zip(peak_im1.frequencies, expected_freqs):
            self.assertAlmostEqual(actual, expected, 4)

    def test_peaks2(self):
        im1 = self.data_for_test1()
        (dists, peaks) = xsp.peakdet.image_peakdet(im1, 0.9)
        expected_dists = [ 2, 4.5 ]
        expected_peaks = [ 1, 10 ]

        for (actual, expected) in zip(dists, expected_dists):
            self.assertAlmostEqual(actual, expected, 4)

        for (actual, expected) in zip(peaks, expected_peaks):
            self.assertAlmostEqual(actual, expected, 4)

    def test_peaks_im2(self):
        im1 = self.data_for_test1()
        peak_im1 = xsp.peakdet.image_as_peaks(im1, 0.9)
        expected_dists = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7,
                            7.5, 8, 8.5, 9, 9.5, 10]
        expected_freqs = [0, 0, 1, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0]

        for (actual, expected) in zip(peak_im1.distances, expected_dists):
            self.assertAlmostEqual(actual, expected, 4)

        for (actual, expected) in zip(peak_im1.frequencies, expected_freqs):
            self.assertAlmostEqual(actual, expected, 4)

if __name__ == '__main__':
    unittest.main()

