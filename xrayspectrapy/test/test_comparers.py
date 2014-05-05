import unittest
import xrayspectrapy as xsp

class ImageComprarerTests(unittest.TestCase):
    def data_for_test1(self):
        return xsp.Image([2, 2.5, 3, 3.5, 4, 4.5, 5],
                [0.5, 0.4, 0.3, 0.8, 0.9, 0.1, 0.1])

    def data_for_test2(self):
        return xsp.Image([2, 2.5, 3, 3.5, 4, 4.5, 5],
                [0.1, 0.3, 0.2, 0.9, 0.0, 0.2, 0.3])

    def test_l2_norm(self):
        im1 = self.data_for_test1()
        im2 = self.data_for_test2()
        dist = xsp.comparers.l2_norm(im1, im2)
        self.assertAlmostEqual(1.0247, dist, 4)

    def test_str(self):
        a = xsp.Atom(5, 8, 9)
        self.assertEqual(str(a), '(5, 8, 9)')

if __name__ == '__main__':
    unittest.main()

