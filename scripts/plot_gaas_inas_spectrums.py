import xrayspectrapy as xsp

files = ('CalcGaAs.dat', 'CalcInAs.dat', 'ExptGaAs.dat', 'ExptInAs.dat')

for afile in files:
    filepath = os.path.expanduser('~/work/gaas_inas_rfdata/' + afile)
    (directory, filename) = os.path.split(filepath)
    (filename, _) = os.path.splitext(filename)
    imgFilePath = os.path.join(directory, filename + '.jpg')
    im = xsp.datadefs.image.fromFile(filepath)
    xsp.datadefs.image.saveAsSpectrumImage(imgFilePath, im, 50, 20)

