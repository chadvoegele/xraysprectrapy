import sys, os
parentdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parentdir)

import xrayspectrapy as xsp

atom_data = ((0.375,0.375,0.375,"Si1"),
             (0.125,0.125,0.625,"Si2"),
             (0.375,0.875,0.875,"Si3"),
             (0.125,0.625,0.125,"Si4"),
             (0.875,0.375,0.875,"Si5"),
             (0.625,0.125,0.125,"Si6"),
             (0.875,0.875,0.375,"Si7"),
             (0.625,0.625,0.625,"Si8"))
l = 5.46872795719
s = xsp.Structure([xsp.Atom(x*l, y*l, z*l, label)
                    for (x, y, z, label) in atom_data],
                    l)

bins = xsp.pdf.calc_bins(1.92, 7.04, 128)
#distancesWLabels = xsp.pdf.calc_distances_with_repetition(s, 1, True)
#print("\n".join([str(x) for x in distancesWLabels]))

pdf = xsp.pdf.calc_pdf(s, 10, bins)
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
