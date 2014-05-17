import xrayspectrapy as xsp
import numpy as np
import matplotlib.mlab as ml

def matToStr(mat):
    return '\n'.join([' '.join(['%.8f' % y for y in x]) for x in mat])

sipath = os.path.expanduser('~/work/rfdata_smoothed/')
sifiles = [os.path.join(sipath, f) for f in os.listdir(sipath)]
sifiles = [f for f in sifiles if os.path.isfile(f)]
siimages = [xsp.datadefs.image.fromFile(f, 'sili-') for f in sifiles]

aspath = os.path.expanduser('~/work/gaas_inas_rfdata/')
asfiles = [os.path.join(aspath, f) for f in os.listdir(aspath)]
asfiles = [f for f in asfiles if os.path.isfile(f)]
asimages = [xsp.datadefs.image.fromFile(f, 'as-') for f in asfiles]

allim = siimages + asimages

datamat = np.array([f.frequencies for f in allim])
# print(matToStr(datamat))

pcamat = ml.PCA(datamat)
# n_pca = 1
# rebuild_datamat = np.dot(pcamat.Y[:,0:n_pca], pcamat.Wt[0:n_pca,:]) * pcamat.sigma\
                    # + pcamat.mu
# print(matToStr(rebuild_datamat))
import pdb; pdb.set_trace()
print(pcamat.mu)
