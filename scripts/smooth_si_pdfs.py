import sys, os
parentdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parentdir)

import xrayspectrapy as xsp

import pdb; pdb.set_trace()
sipath = os.path.expanduser('~/work/rfdata/')
sifiles = [os.path.join(sipath, f) for f in os.listdir(sipath)]
sifiles = [f for f in sifiles if os.path.isfile(f)]
siimages = [xsp.datadefs.image.fromFile(f) for f in sifiles]
siimages = [im for im in siimages if "Calc" in im.label]
siimages = [xsp.pdf.smooth_image(im, 0.004) for im in siimages]

sipathout = os.path.expanduser('~/work/rfdata_smoothed')
os.mkdir(sipathout)
[xsp.datadefs.image.toFile(os.path.join(sipathout, im.label + '.dat'), im)
        for im in siimages]
