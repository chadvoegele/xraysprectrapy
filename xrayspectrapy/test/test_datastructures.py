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

if __name__ == '__main__':
    unittest.main()
