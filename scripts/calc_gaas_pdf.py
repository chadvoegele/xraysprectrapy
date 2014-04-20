import sys, os
parentdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parentdir)

import xrayspectrapy as xsp

s = xsp.Structure([xsp.Atom(0.00,0.00,0.00,"Ga1"),
                    xsp.Atom(0.00,0.50,0.50,"Ga2"),
                    xsp.Atom(0.50,0.00,0.50,"Ga3"),
                    xsp.Atom(0.50,0.50,0.00,"Ga4"),
                    xsp.Atom(0.25,0.25,0.25,"As1"),
                    xsp.Atom(0.75,0.75,0.25,"As2"),
                    xsp.Atom(0.75,0.25,0.75,"As3"),
                    xsp.Atom(0.25,0.75,0.75,"As4")])

c = 5.6536998749
#distancesWLabels = xsp.pdf.calc_distances_with_repetition(s, 10/c, True)
#distancesWLabels = [xsp.pdf.DistancesWithLabels(c*d.distance, d.label1, d.label2)
#    for d in distancesWLabels]
#print("\n".join([str(x) for x in distancesWLabels]))


bins = xsp.pdf.calc_bins(1, 10, 200)
bins = [b / c for b in bins]
im = xsp.pdf.calc_pdf(s, 10/c, bins)
sf = sum(im.frequencies)
im = xsp.Image([d * c for d in im.distances], 
        [f / sf for f in im.frequencies])
im = xsp.pdf.smooth_image(im, 0.005)
print(im)
