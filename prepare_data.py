import h5py
import numpy
import cv2
import sys
import argparse

def main(path, output, stride, patch, random=False):

    count = 0

    print('Data will be at {} with stride {}, with patch size {}.'.format(path, stride, patch))

    rand = numpy.random.randint(2)
    if rand == 0:
        with h5py.File(path + 'recon_even.h5', 'r') as fd:
            f = numpy.array(fd['images'])
    else:
        with h5py.File(path + 'recon_odd.h5', 'r') as fd:
            f = numpy.array(fd['images'])
            
    with h5py.File(path + 'recon.h5' , 'r') as gd:
        g = numpy.array(gd['images'])

    f_rescale = numpy.empty((f.shape[0]*2, f.shape[1], f.shape[2]*2))

    for i in range(f.shape[1]):
        f_rescale[:,i,:] = cv2.resize(f[:,i,:], (f.shape[0]*2, f.shape[2]*2), interpolation=cv2.INTER_CUBIC) 
    # f_rescale = cv2.normalize(f_rescale, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    f = ((f_rescale - f_rescale.min()) * 1.0000000 / (f_rescale.max() - f_rescale.min()))
    g = ((g - g.min()) * 1.0000000 / (g.max() - g.min()))
    del(f_rescale)

    print('Data shape: {}'.format(f.shape))
    print('Label shape: {}'.format(g.shape))

    f_shape = f.shape
    g_shape = g.shape

    # total_patches = ((shape[1]-patch)//stride+1)*((shape[0]-patch)//stride+1)*((shape[2]-patch)//stride+1)

    f_total_patches = (((f_shape[0] - patch)//stride) + 1)**2
    f_total_patches *= f_shape[1]

    g_total_patches = (((g_shape[0] - patch)//stride) + 1)**2
    g_total_patches *= g_shape[1]

    print('Total data patches {}'.format(f_total_patches))
    print('Total label patches {}'.format(g_total_patches))

    data = numpy.empty((int(f_total_patches), patch, patch, 1))
    label = numpy.empty((int(g_total_patches), patch, patch, 1))

    for i in range(int(f_shape[1])):
        for j in range(0, f_shape[0] - patch, stride):
            for k in range(0, f_shape[2] - patch, stride):
                data[count, :, :, 0] = f[j:j+patch, i, k:k+patch]
                count+=1

    count = 0
    for i in range(int(g_shape[1])):
        for j in range(0, g_shape[0] - patch, stride):
            for k in range(0, g_shape[2] - patch, stride):
                label[count, :, :, 0] = g[j:j+patch, i, k:k+patch]
                count+=1

    del(f)
    del(g)
    print(data.shape)
    print(label.shape)
    
    if random:
        order = numpy.random.permutation(count)
        label = label[order]
        data = data[order]

    with h5py.File(output, 'w') as h5:
        h5.create_dataset('data', shape=data.shape, dtype='float32', data=data)
        h5.create_dataset('label', shape=label.shape, dtype='float32', data=label)

if __name__ == '__main__':

    count = 0
    parser = argparse.ArgumentParser(description = 'Gather data for the VDSR.')
    parser.add_argument('--path', help='Path where the data is located.', required=True)
    parser.add_argument('-s', '--stride', help='Size of the stride between subpatchs.', required=True, type=int)
    parser.add_argument('-p', '--patch-size', help='Patch size.', required=True, type=int)
    parser.add_argument('-o', '--output-path', help='Path where the data will be saved.', required=True)
    parser.add_argument('-r', '--randomize', help='Wether to randomize data order or not.', action='store_true')

    arguments = parser.parse_args()

    path = arguments.path
    stride = int(arguments.stride)
    patch = int(arguments.patch_size)
    output = arguments.output_path
    random = False
    if arguments.randomize:
        random = True

    main(path, output, stride, patch, random)
