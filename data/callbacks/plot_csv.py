import csv
import matplotlib.pyplot as plt
import sys
import numpy

if len(sys.argv) < 2 or not type(sys.argv[1]) == str:
    sys.exit('Missing arguments.')

fname = sys.argv[1]
epoch = []
ssim = []
psnr = []
val_ssim = []
val_psnr = []

with open(fname, 'r') as csvfile:
    data = csv.DictReader(csvfile)
    for d in data:
        epoch.append(int(d['epoch']))
        ssim.append(float(d['ssim']))
        psnr.append(float(d['PSNR']))
        val_ssim.append(float(d['val_ssim']))
        val_psnr.append(float(d['val_PSNR']))

p = numpy.poly1d(numpy.polyfit(epoch, ssim, 5))
q = numpy.poly1d(numpy.polyfit(epoch, psnr, 5))
r = numpy.poly1d(numpy.polyfit(epoch, val_ssim, 5))
s = numpy.poly1d(numpy.polyfit(epoch, val_psnr, 5))

plt.subplot(223)
plt.plot(epoch, p(epoch), label='Training')
plt.plot(epoch, r(epoch), label='Validation')
plt.subplot(224)
plt.plot(epoch, q(epoch), label='Training')
plt.plot(epoch, s(epoch), label='Validation')

plt.subplot(221)
plt.ylabel('SSIM')
plt.xlabel('EPOCHS')
plt.locator_params(axis='y', nbins=20)
plt.title('SSIM')
plt.plot(epoch, ssim, label='Training')
plt.plot(epoch, val_ssim, label='Validation')
plt.legend()
plt.subplot(222)
plt.ylabel('PSNR')
plt.xlabel('EPOCHS')
plt.locator_params(axis='y', nbins=20)
plt.title('PSNR')
plt.plot(epoch, psnr, label='Training')
plt.plot(epoch, val_psnr, label='Validation')
plt.legend()
plt.show()
