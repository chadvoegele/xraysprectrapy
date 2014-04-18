"""Holds all atoms in chemical structure."""
class Structure:
    def __init__(self, atoms):
        self.atoms = atoms

    def __str__(self):
        out_str = '['
        for atom in self.atoms:
            out_str = out_str + str(atom)
        out_str = out_str + ']'
        return out_str
