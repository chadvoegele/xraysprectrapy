import xrayspectrapy as xsp
from scipy import interpolate

atom_data = [(0.00,0.00,0.00,"In1"),
             (0.00,0.50,0.50,"In1"),
             (0.50,0.00,0.50,"In3"),
             (0.50,0.50,0.00,"In4"),
             (0.25,0.25,0.25,"As1"),
             (0.75,0.75,0.25,"As2"),
             (0.75,0.25,0.75,"As3"),
             (0.25,0.75,0.75,"As4")]
c = 6.0583

s = xsp.Structure([xsp.Atom(c*x, c*y, c*z, l) for (x,y,z,l) in atom_data], c)

bins = xsp.pdf.calc_bins(1.92, 7.04, 128)
im = xsp.pdf.calc_pdf(s, 10, bins)
im = xsp.pdf.smooth_image(im, 0.005)
print(im)

expt_filename = os.path.expanduser('~/work/inas_fig4.csv')
im = xsp.datadefs.image.fromFile(expt_filename)
f = interpolate.interp1d(im.distances, im.frequencies, kind = 'cubic')
freq = [f(y) for y in bins[:-1]]
minfreq = min(freq)
freq = [y - minfreq for y in freq]
sumfreq = sum(freq)
freq = [y/sumfreq for y in freq]
imnew = xsp.Image(bins[:-1], freq)
# print(imnew)
