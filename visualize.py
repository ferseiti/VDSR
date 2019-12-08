import matplotlib.pyplot as plt, numpy; from tifffile import imread

a = numpy.array(imread('data/images/data/images/001.tif')); b = numpy.array(imread('data/imagesHR/001.tif')); c = numpy.array(imread('data/images/001.tif'))
a = ((a - a.min()) * 255.0 / (a.max() - a.min())); b = ((b - b.min()) * 255.0 / (b.max() - b.min())); c = ((c - c.min()) * 255.0 / (c.max() - c.min()))
plt.subplot(221);plt.imshow(a[10:-10,10:-10]);plt.subplot(222),plt.imshow(b); plt.subplot(223),plt.imshow(c);plt.show()
