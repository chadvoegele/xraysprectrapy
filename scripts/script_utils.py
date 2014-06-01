import xrayspectrapy as xsp

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
    else if type(data) is int:
        return '%d' % data
    else if type(data) is float:
        return '%.8f' % data
    else
        return str(data)

