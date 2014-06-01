import xrayspectrapy as xsp
import os

def plotImages(outdir, images, filePrefix, fileExt, legendLoc=1):
    directory = os.path.expanduser(outdir)
    name = '-'.join([im.label for im in images])
    filename = os.path.join(directory, filePrefix + name + '.' + fileExt)
    xsp.datadefs.image.saveAllAsLineImages(filename, images, legendLoc=legendLoc)

def getAllImages(filedir, filterStrs):
    calcFiles = [os.path.join(filedir, f) for f in os.listdir(filedir)
                   if os.path.isfile(os.path.join(filedir, f))
                   if any((s in f for s in filterStrs))]
    calcImages = [xsp.datadefs.image.fromFile(f) for f in calcFiles]
    return calcImages

def matToStr(mat):
    return '\n'.join(['\t'.join([matDataToStr(y) for y in x]) for x in mat])

def matDataToStr(data):
    if type(data) is str:
        return data
    elif type(data) is int:
        return '%d' % data
    elif type(data) is float:
        return '%.8f' % data
    else:
        return str(data)

