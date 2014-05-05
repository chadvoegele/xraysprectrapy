import sys, os
parentdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parentdir)

import xrayspectrapy as xsp

atom_data = [(1.2455,0.5367,-0.0729,"C1"),
             (0.9239,-0.9952,0.0237,"C2"),
             (-0.1226,-0.7041,1.1548,"C3"),
             (0.1989,0.8277,1.0582,"C4"),
             (0.1226,0.7042,-1.1548,"C5"),
             (-0.9239,0.9952,-0.0237,"C6"),
             (-1.2454,-0.5367,0.0729,"C7"),
             (-0.1989,-0.8277,-1.0582,"C8"),
             (2.2431,0.9666,-0.1313,"H1"),
             (1.6638,-1.7924,0.0426,"H2"),
             (-0.2209,-1.2683,2.0797,"H3"),
             (0.3583,1.4907,1.9059,"H4"),
             (0.2208,1.2681,-2.0799,"H5"),
             (-1.6640,1.7922,-0.0427,"H6"),
             (-2.2430,-0.9665,0.1313,"H7"),
             (-0.3583,-1.4906,-1.9058,"H8")]

s = xsp.Structure([xsp.Atom(x, y, z, l) for (x,y,z,l) in atom_data])

bins = xsp.pdf.calc_bins(1, 10, 200)
im = xsp.pdf.calc_pdf(s, 10, bins)
im = xsp.pdf.smooth_image(im, 0.005)
print(im)