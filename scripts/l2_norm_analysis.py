import xrayspectrapy as xsp
import math
import numpy as np

filedir = os.path.expanduser('~/work/all_rfdata_smoothed_unique/')
exptFiles = [os.path.join(filedir, f) for f in 
                ('ExptGaAs.dat', 'ExptInAs.dat', 'SiLiExpt1.dat')]
allCalcFiles = [os.path.join(filedir, f) for f in os.listdir(filedir)
               if os.path.isfile(os.path.join(filedir, f))
               if ('Calc' in f or 'calc' in f)]

exptImages = [xsp.datadefs.image.fromFile(f) for f in exptFiles]
allCalcImages = [xsp.datadefs.image.fromFile(f) for f in allCalcFiles]

labels = [im.label for im in allCalcImages]
for exptImage in exptImages:
    dists = [xsp.comparers.l2_norm(exptImage, calcImg)
                for calcImg in allCalcImages]
    print(exptImage.label + "->" + labels[np.argmin(dists)])

