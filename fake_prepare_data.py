import h5py
import numpy
import cv2
import sys
import argparse

def main(path, output, stride, patch, random=False):

    count = 0
    SCALE = 2
    print('Data will be at {} with stride {}, with patch size {}.'.format(path, stride, patch))

    with h5py.File(path + 'recon.h5' , 'r') as gd:
        g = numpy.array(gd['images'])

    # for i in range(f.shape[1]):
    #     # f_rescale[:,i,:] = cv2.resize(f[:,i,:], (f.shape[0]*2, f.shape[2]*2), interpolation=cv2.INTER_CUBIC)
    # # f_rescale = cv2.normalize(f_rescale, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # # f = f_rescale

    print('Data shape: {}'.format(g.shape))
    print('Label shape: {}'.format(g.shape))

    g_shape = g.shape

    # total_patches = ((shape[1]-patch)//stride+1)*((shape[0]-patch)//stride+1)*((shape[2]-patch)//stride+1)

    # f_total_patches = (((f_shape[0] - patch)//stride) + 1)**2
    # f_total_patches *= f_shape[1]

    g_total_patches = (((g_shape[0] - patch - 500)//stride) + 1)**2
    g_total_patches *= g_shape[1]
    f_total_patches = g_total_patches

    print('Total data patches {}'.format(f_total_patches))
    print('Total label patches {}'.format(g_total_patches))

    data = numpy.empty((int(f_total_patches), patch, patch, 1))
    label = numpy.empty((int(g_total_patches), patch-4, patch-4, 1))

    import warnings
    numpy.seterr(all='raise')
    warnings.filterwarnings('error')

    found = False
    for i in range(int(g_shape[1])):
        if not found:
            print('Not found yet')
            for j in range(250, g_shape[0] - patch - 250, stride):
                if not found:
                    print('Not found yet')
                    for k in range(250, g_shape[2] - patch - 250, stride):
                        if not (g[j:j+patch, i, k:k+patch].max() - g[j:j+patch, i, k:k+patch].min() == 0.0):
                            good_data = cv2.resize(g[j:j+patch:SCALE, i, k:k+patch:SCALE], (patch,patch), interpolation=cv2.INTER_CUBIC)
                            good_label = g[2+j:j+patch-2, i, 2+k:k+patch-2]
                            good_data = (good_data - good_data.min()) * 1.0000000 / (good_data.max() - good_data.min())
                            good_label = (good_label - good_label.min()) * 1.0000000 / (good_label.max() - good_label.min())
                            found = True
                            break
                else:
                    break
        else:
            break
    count = 0
    
    for i in range(int(g_shape[1])):
        for j in range(250, g_shape[0] - patch - 250, stride):
            for k in range(250, g_shape[2] - patch - 250, stride):
                data[count, :, :, 0] = cv2.resize(g[j:j+patch:SCALE, i, k:k+patch:SCALE], (patch,patch), interpolation=cv2.INTER_CUBIC)
                label[count, :, :, 0] = g[2+j:j+patch-2, i, 2+k:k+patch-2]
                try:
                    data[count, :, :, 0] = ((data[count, :, :, 0] - data[count, :, :, 0].min()) * 1.0000000 / (data[count, :, :, 0].max() - data[count, :, :, 0].min()))
                    label[count, :, :, 0] = ((label[count, :, :, 0] - label[count, :, :, 0].min()) * 1.0000000 / (label[count, :, :, 0].max() - label[count, :, :, 0].min()))
                except:
                    print(path)
                    print((data[count, :, :, 0].max() - data[count, :, :, 0].min()), (label[count, :, :, 0].max() - label[count, :, :, 0].min()))
                    data[count, :, :, 0] = good_data
                    label[count, :, :, 0] = good_label
                    print('This used the good one: {}'.format(count))
                count += 1
    print('last count: {}'.format(count))

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
