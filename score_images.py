import os
import numpy
from tifffile import imread
from keras import backend as K

from inception_score import get_inception_score

def grayscale_to_rgb(images, channel_axis=-1):
    images = K.expand_dims(images, axis=channel_axis)
    tiling = [1] * 4    # 4 dimensions: B, H, W, C
    tiling[channel_axis] *= 3
    images = K.tile(images, tiling)
    return images

def score(path):
    print("--------------------------------")
    print('path:', path)
    print("--------------------------------")

    li = os.listdir(path)
    imgs = []
    for filename in li:
        if '.tif' in filename:
            img = numpy.array(imread(os.path.join(path, filename)))
            img = img.reshape(img.shape+(1,))
            img = ((img - img.min()) * 255.0 / (img.max() - img.min()))
            imgs.append(img)
        else:
            pass
    try:
        imgs = grayscale_to_rgb(imgs)
    except:
        print('Not able to convert to rgb')
    # print(imgs.shape)
    print('min : ', numpy.min(imgs[0]))
    print('max : ', numpy.max(imgs[0]))

    result = get_inception_score(imgs)

    with open('%s/inception_score.txt' % path, 'w') as f:
        f.write('inception score...\n')
        f.write('mean : %s\n' % result[0])
        f.write('std : %s\n' % result[1])
        f.close()

    print('scoring finished!')
    print('mean : %s\n' % result[0])
    print('std : %s\n' % result[1])
