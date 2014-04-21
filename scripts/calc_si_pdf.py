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
#distancesWLabels = xsp.pdf.calc_distances_with_repetition(s, 1, True)
#print("\n".join([str(x) for x in distancesWLabels]))

pdf = xsp.pdf.calc_pdf(s, 10/c, [a / c for a in bins])
freq_sum = sum(pdf.frequencies)
pdf.frequencies = [a / freq_sum for a in pdf.frequencies]
pdf.distances = [a * c for a in pdf.distances]
my_calc_pdf = xsp.pdf.smooth_image(pdf, 0.004)

actualFilename = os.path.expanduser('~/work/rfdata/Calc10001.txt')
actual_calc_pdf = xsp.datadefs.image.fromFile(actualFilename)
actual_calc_pdf = xsp.pdf.smooth_image(actual_calc_pdf, 0.004)

exptFilename = os.path.expanduser('~/work/rfdata/Expt1.txt')
expt_pdf = xsp.datadefs.image.fromFile(exptFilename)

#gaussian smoothing calibration
#ts = [x*0.0001 for x in range(25, 50)]
#errors = [xsp.comparers.least_squares(
#            xsp.pdf.smooth_image(actual_calc_pdf, t), expt_pdf)
#        for t in ts]
#print("\n".join([str(t) + '\t' + str(error) for (t, error) in zip(ts, errors)]))

print(my_calc_pdf)
