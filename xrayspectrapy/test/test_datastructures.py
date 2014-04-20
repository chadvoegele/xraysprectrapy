import unittest
import xrayspectrapy as xsp

class AtomTests(unittest.TestCase):
    def test_str(self):
        a = xsp.Atom(5, 8, 9)
        self.assertEqual(str(a), '(5, 8, 9)')

class StructureTests(unittest.TestCase):
    def test_str(self):
        s = xsp.Structure([xsp.Atom(5, 8, 9), xsp.Atom(5, 2, 0)])
        self.assertEqual(str(s), '[(5, 8, 9)(5, 2, 0)]')

class ImageTests(unittest.TestCase):
    def test_str(self):
        x = xsp.Image([0.24, 0.28, 0.32, 0.36], [.0001, 0.50, 0.449, 0.1112])
        self.assertEqual(str(x),
                'Image: \n0.24\t0.0001\n0.28\t0.5\n0.32\t0.449\n0.36\t0.1112')

    def test_from_file(self):
        expectedDists = [1.92, 1.96, 2.00, 2.04]
        expectedFrequencies = [0.0005, 0.0010, 0.0015, 0.0205]
        im = xsp.datadefs.image.fromFile("xrayspectrapy/test/test_image.txt")

        for i in range(0,len(expectedDists)):
            self.assertAlmostEqual(expectedDists[i], im.distances[i], 2)

        for i in range(0,len(expectedFrequencies)):
            self.assertAlmostEqual(expectedFrequencies[i], im.frequencies[i], 4)


if __name__ == '__main__':
    unittest.main()
