"""Holds all atoms in chemical structure."""
class Structure:
    def __init__(self, atoms, period_length=None):
        self.atoms = atoms
        self.period_length = period_length

    def __str__(self):
        out_str = '['
        for atom in self.atoms:
            out_str = out_str + str(atom)
        out_str = out_str + ']'
        if self.period_length != None:
            out_str = out_str + "Length: " + period_length
        return out_str
