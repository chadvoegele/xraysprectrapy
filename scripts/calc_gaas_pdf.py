import sys, os
parentdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parentdir)

import xrayspectrapy as xsp
from scipy import interpolate

atom_data = [(0.00,0.00,0.00,"Ga1"),
            (0.00,0.50,0.50,"Ga2"),
            (0.50,0.00,0.50,"Ga3"),
            (0.50,0.50,0.00,"Ga4"),
            (0.25,0.25,0.25,"As1"),
            (0.75,0.75,0.25,"As2"),
            (0.75,0.25,0.75,"As3"),
            (0.25,0.75,0.75,"As4")]
c = 5.6536998749

s = xsp.Structure([xsp.Atom(c*x, c*y, c*z, l) for (x,y,z,l) in atom_data], c)

#distancesWLabels = xsp.pdf.calc_distances_with_repetition(s, 10/c, True)
#distancesWLabels = [xsp.pdf.DistancesWithLabels(c*d.distance, d.label1, d.label2)
#    for d in distancesWLabels]
#print("\n".join([str(x) for x in distancesWLabels]))

bins = xsp.pdf.calc_bins(1.92, 7.04, 128)
im = xsp.pdf.calc_pdf(s, 10, bins)
im = xsp.pdf.smooth_image(im, 0.005)
# print(im)

expt_filename = os.path.expanduser('~/work/gaas_fig4.csv')
im = xsp.datadefs.image.fromFile(expt_filename)
f = interpolate.interp1d(im.distances, im.frequencies, kind = 'cubic')
freq = [f(y) for y in bins[:-1]]
minfreq = min(freq)
freq = [y - minfreq for y in freq]
sumfreq = sum(freq)
freq = [y/sumfreq for y in freq]
im = xsp.Image(bins[:-1], freq)
print(im)
