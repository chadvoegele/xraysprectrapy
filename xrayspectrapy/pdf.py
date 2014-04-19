import numpy as np
import xrayspectrapy as xsp

"""Calculations pertaining to pair distribution function."""

def calc_pdf(structure, maxOffset = 0, binsOrnBins = 128):
    """Calculates the pair distribution function for structure.

       Arguments: structure -- xrayspectrapy.structure.Structure
                  maxOffset -- int
                  binsOrnBins -- int or sequence

       Example: calc_pdf(structure, 1, [0 0.2 0.4 0.6 0.8 1])
                calc_pdf(structure, 1, 5)
    """
    distances = calc_distances_with_repetition(structure, maxOffset)

    if isinstance(binsOrnBins, int):
        nBins = binsOrnBins
        bins = calc_bins(round(min(distances), 1), round(max(distances), 1), nBins)
    else:
        bins = binsOrnBins

    (frequencies, _) = np.histogram(distances, bins)
    return xsp.Image(bins[:-1], frequencies)

def calc_bins(lowerBnd, upperBnd, nBins):
    """Returns nBins number of intervals equally spaced between lowerBnd and
       upperBnd.

       Example: calc_bins(0.2, 1, 4) -> [.2 .4 .6 .8 1]
       
    """
    increment = (upperBnd - lowerBnd) / nBins
    return [x * increment + lowerBnd for x in range(0, nBins + 1)]

def calc_distances(structure):
    """Calculate all pairwise distances for atoms in structure.
    
       Arguments:
       structure -- xrayspectrapy.structure.Structure
       maxOffset -- int
    """
    zeroOffset = np.array((0, 0, 0))
    return calc_offset_distances(structure, zeroOffset, zeroOffset)

def calc_distances_with_repetition(structure, nPeriods, wantLabels = False):
    """Calculates pairwise distances where structure is repeated nPeriods times
        in all directions. This only works if coordinates of the structures are
        periodic with a distance of periodLength in all directions.

       In 2D for nPeriods = 1,
          --------------------------
          |        |        |      |
       (-1,1)----(0,1)----(1,1)-----
          |        |        |      |
       (-1,0)----(0,0)----(1,0)-----
          |        |        |      |
       (-1,-1)---(0,-1)---(1,-1)----

       Arguments:
       structure -- XRaySpectraLib.structure.Structure
       maxOffset -- int
    """
    allCoords = [x for atom in structure.atoms 
                    for x in [atom.x, atom.y, atom.z]]
    if (not (min(allCoords) >= 0 and max(allCoords) <= 1)):
        raise Exception('min coord must not be less than 0 and max coord must' +
                           ' not be more than 1')

    ind = range(-nPeriods, nPeriods + 1)  # [-nPeriods, nPeriods]
    offsets = [np.array((i, j, k)) for i in ind for j in ind for k in ind]
    pairOffsets = [(x, y) for x in offsets for y in offsets]
    
    return [d for (o1, o2) in pairOffsets
              for d in calc_offset_distances(structure, o1, o2, wantLabels)]

def calc_offset_distances(structure, offset1, offset2, wantLabels = False):
    """Calculates all pairwise distances for atoms in structure where one of the
        atoms has been displaced by offset1 and the other by offset2.
       For all i, j where i != j, dist(atom_i + offset1, atom_j + offset2)
    """
    atoms = structure.atoms
    nAtoms = len(atoms)
    pairIndices = [(i,j) for i in range(0, nAtoms) for j in range(0, nAtoms)]
    return [calc_distance(atoms[i], atoms[j], offset1, offset2, wantLabels)
            for (i, j) in pairIndices]

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
