import matplotlib.pyplot as plt, numpy; from tifffile import imread
import cv2

B=1
I="{0:0=3d}".format(numpy.random.randint(21))
print(I)

a = numpy.array(imread('data/images/data/images/{}.tif'.format(I))); b = numpy.array(imread('data/imagesHR/{}.tif'.format(I))); c = numpy.array(imread('data/images/{}.tif'.format(I)))
#a = numpy.array(imread('0001/lr/0001/lr/recon/020.tif')); b = numpy.array(imread('0001/hr/020.tif')); c = numpy.array(imread('0001/lr/020.tif'))
a *= -1
d = cv2.resize(c, (2048, 2048), interpolation=cv2.INTER_CUBIC)
a = ((a - a.min()) * 1.0000 / (a.max() - a.min())); b = ((b - b.min()) * 1.0000 / (b.max() - b.min())); c = ((c - c.min()) * 1.0000 / (c.max() - c.min()))

a1 = a[B:-B,B:-B]
print(a1.shape)
plt.subplot(221)
plt.title('VDSR')
#plt.imshow(a[980:1100,1600:1740])
plt.imshow(a1)
plt.subplot(222)
plt.title('HR')
#plt.imshow(b[980:1100,1600:1740])
plt.imshow(b)
plt.subplot(223)
plt.title('LR')
#plt.imshow(c[490:550,800:870])
plt.imshow(c)
plt.subplot(224)
plt.title('Bicubic')
#plt.imshow(d[980:1100,1600:1740])
plt.imshow(d)


plt.show()
