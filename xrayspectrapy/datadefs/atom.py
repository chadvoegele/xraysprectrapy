"""Holds atom coordinate data."""
class Atom:
    def __init__(self, x, y, z, label=""):
        self.x = x
        self.y = y
        self.z = z
        self.label = label

    def __str__(self):
        return '(' + str(self.x) + ', ' + str(self.y) + ', ' + str(self.z) + ')'
