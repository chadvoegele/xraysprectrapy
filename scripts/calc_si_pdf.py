import sys, os
parentdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parentdir)

import xrayspectrapy as xsp

s = xsp.Structure([xsp.Atom(0.375,0.375,0.375,"Si1"),
                   xsp.Atom(0.125,0.125,0.625,"Si2"),
                   xsp.Atom(0.375,0.875,0.875,"Si3"),
                   xsp.Atom(0.125,0.625,0.125,"Si4"),
                   xsp.Atom(0.875,0.375,0.875,"Si5"),
                   xsp.Atom(0.625,0.125,0.125,"Si6"),
                   xsp.Atom(0.875,0.875,0.375,"Si7"),
                   xsp.Atom(0.625,0.625,0.625,"Si8")])

c = 5.46872795719
bins = xsp.pdf.calc_bins(1.92, 7.04, 128)
bins = [a / c for a in bins]

#distancesWLabels = xsp.pdf.calc_distances_with_repetition(s, 1, True)
#print("\n".join([str(x) for x in distancesWLabels]))

pdf = xsp.pdf.calc_pdf(s, 2, bins)
freq_sum = sum(pdf.frequencies)
pdf.frequencies = [a / freq_sum for a in pdf.frequencies]
pdf.distances = [a * c for a in pdf.distances]

print(pdf)
