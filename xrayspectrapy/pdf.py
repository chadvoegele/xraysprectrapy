import numpy as np

"""Calculations pertaining to pair density function."""

def calc_pdf(structure):
    return calc_pdf(structure, 0)

def calc_pdf(structure, maxOffset):
    distances = calc_distances_with_repetition(structure, maxOffset)
    nBins = 50
    lowerBnd = round(min(distances), 2)
    upperBnd = round(max(distances), 2)
    increment = (upperBnd - lowerBnd) / nBins
    bins = [x * increment + lowerBnd for x in range(0, nBins)]
    (frequencies, _) = np.histogram(distances, bins)

def calc_distances(structure):
    """Calculate all pairwise distances for atoms in structure.
    
       Arguments:
       structure -- XRaySpectraLib.structure.Structure
    """
    zeroOffset = np.array((0, 0, 0))
    return calc_distances_with_offsets(structure, zeroOffset, zeroOffset)

def calc_distances_with_repetition(structure, maxOffset):
    """Calculates pairwise distances where structure is repeated maxOffset times
        in all directions. This only works if coordinates of the structures are
        periodic with a distance of 1 in all directions.

       In 2D for maxOffset = 1,
       (-1,1)----(0,1)----(1,1)
          |        |        |
       (-1,0)----(0,0)----(1,0)
          |        |        |
       (-1,-1)---(0,-1)---(1,-1)

       Arguments:
       structure -- XRaySpectraLib.structure.Structure
       maxOffset -- int
    """
    allCoords = [x for atom in structure.atoms 
                    for x in [atom.x, atom.y, atom.z]]
    if (not (min(allCoords) >= 0 and max(allCoords) <= 1)):
        raise Exception('min coord must not be less than 0 and max coord must' +
                           ' not be more than 1')

    ind = range(-maxOffset, maxOffset)  # [-maxOffset, maxOffset-1]
    offsets = [np.array((i, j, k)) for i in ind for j in ind for k in ind]
    pairOffsets = [(x, y) for x in offsets for y in offsets]
    
    return [d for (offset1, offset2) in pairOffsets
              for d in calc_distances_with_offsets(structure, offset1, offset2)]

def calc_distances_with_offsets(structure, offset1, offset2):
    """Calculates all pairwise distances for atoms in structure where one of the
        atoms has been displaced by offset1 and the other by offset2.
       For all i, j where i != j, dist(atom_i + offset1, atom_j + offset2)
    """
    atoms = structure.atoms
    nAtoms = len(atoms)
    pairIndices = [(i,j) for i in range(0, nAtoms) for j in range(0, nAtoms) 
            if i != j]
    return [calc_distance(atoms[i], atoms[j], offset1, offset2)
            for (i, j) in pairIndices]

def calc_distance(atom1, atom2, offset1, offset2):
    """Calculates distance between atom1 and atom2 where atom1 is offset by
        offset1 and atom2 if offset by offset2.
       dist(atom2 + offset2, atom1 + offset1)

       Arguments:
       atom1, 2 -- XRaySpectraLib.atom.Atom
       offset1, 2 -- numpy.ndarray
    """
    a1 = np.array((atom1.x, atom1.y, atom1.z))
    a2 = np.array((atom2.x, atom2.y, atom2.z))
    offsetAtomsDifference = a2 + offset2 - (a1 + offset1)
    return np.sqrt(np.inner(offsetAtomsDifference, offsetAtomsDifference))

