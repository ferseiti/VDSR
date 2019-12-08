import h5py
import numpy
import cv2
import sys
import argparse

def main(path, output, stride, patch, random=False, interpolate=False, sample=1.0):

    count = 0

    print('Data will be at {} with stride {}, with patch size {}.'.format(path, stride, patch))

    with h5py.File(path, 'r') as fd:
        f = numpy.array(fd['images'])

    if interpolate:
        f_rescale = numpy.empty((f.shape[0]*2, f.shape[1], f.shape[2]*2))

        for i in range(f.shape[1]):
            f_rescale[:,i,:] = cv2.resize(f[:,i,:], (f.shape[0]*2, f.shape[2]*2), interpolation=cv2.INTER_CUBIC) 
        # f_rescale = cv2.normalize(f_rescale, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        f = cv2.normalize(numpy.array(f_rescale), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    print(f.shape)

    shape = f.shape

    # total_patches = ((shape[1]-patch)//stride+1)*((shape[0]-patch)//stride+1)*((shape[2]-patch)//stride+1)

    total_patches = (((shape[0] - patch)//stride) + 1)**2
    total_patches *= shape[1]

    print('Total patches {}'.format(total_patches))

    data = numpy.empty((int(sample*total_patches), patch, patch, 1))

    for i in range(int(shape[1]*sample)):
        for j in range(0, shape[0] - patch, stride):
            for k in range(0, shape[2] - patch, stride):
                data[count, :, :, 0] = f[j:j+patch, i, k:k+patch]
                count+=1
    
    print(data.shape)
    
    if random:
        order = numpy.random.permutation(count)
        label = label[order]
        data = data[order]

    with h5py.File(output, 'w') as h5:
        h5.create_dataset('data', shape=data.shape, dtype='float32', data=data)

if __name__ == '__main__':

    count = 0
    parser = argparse.ArgumentParser(description = 'Gather data for the 3DSRCNN.')
    parser.add_argument('-f', '--file', help='Path where the data is located.', required=True)
    parser.add_argument('-s', '--stride', help='Size of the stride between subpatchs.', required=True, type=int)
    parser.add_argument('-p', '--patch-size', help='Patch size.', required=True, type=int)
    parser.add_argument('-o', '--output-path', help='Path where the data will be saved.', required=True)
    parser.add_argument('-r', '--randomize', help='Wether to randomize data order or not.', action='store_true')
    parser.add_argument('-i', '--interpolate', help='Wether to interpolate data or not.', action='store_true')
    parser.add_argument('--sample', help='Sample amount (fraction).', required=False, type=float, default=1.0)


    arguments = parser.parse_args()

    path = arguments.file
    stride = int(arguments.stride)
    patch = int(arguments.patch_size)
    output = arguments.output_path
    random = False
    interpolate = False
    sample = arguments.sample
    if arguments.randomize:
        random = True
    if arguments.interpolate:
        interpolate = True

    main(path, output, stride, patch, random, interpolate, sample)
