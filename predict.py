import sys
import os
from keras.models import model_from_json, Model
from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, Add, Subtract, Dense, Activation, add, merge
import numpy
import cv2

from tifffile import imread, imsave

# from scipy.misc import toimage, imread, imresize
import argparse

from score_images import score

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Predict image super resolution with VDSR.')
    parser.add_argument('-j', '--json-path', help='Path where the model json is located.', required=True)
    parser.add_argument('-w', '--weights-path', help='Path where the model is located.', required=True)
    parser.add_argument('-i', '--image-path', help='Path where the image is located.', required=True)
    parser.add_argument('-o', '--output-path', help='Path where the output image will be saved.', required=True)
    parser.add_argument('-s', '--size', help='Size (width) of the square image', required=True, type=int)

    arguments = parser.parse_args()

    json_path = arguments.json_path
    w_path = arguments.weights_path
    img_path = arguments.image_path
    dst_path = arguments.output_path
    target_size = (arguments.size, arguments.size) # should be changed according to size of output image
    print("--------------------------------")
    print('json_path : ', json_path)
    print('w_path : ', w_path)
    print('img_path : ', img_path)
    print('dst_path : ', dst_path)
    print("--------------------------------")

    input_img = Input(shape=target_size+(1,))

    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)
    model_0 = Activation('relu')(model)

    total_conv = 22  # should be even number
    total_conv -= 2  # subtract first and last
    residual_block_num = 5  # should be even number

    for _ in range(residual_block_num):  # residual block
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model_0)
        model = Activation('relu')(model)
        for _ in range(int(total_conv/residual_block_num)-1):
            model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
            model = Activation('relu')(model)
            model_0 = add([model, model_0])

    model = Conv2D(1, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    res_img = model
    output_img = merge.Subtract()([res_img, input_img])

    model = Model(input_img, output_img)

    vdsr = model

    li = os.listdir(img_path)

    target_path = '%s/%s/' % (img_path, dst_path)
    os.makedirs(target_path, exist_ok=True)
    for filename in li:
        if '.tif' in filename:
            img = imread(os.path.join(img_path, filename))
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC) 
            # img = imresize(img, target_size, interp='bicubic')
            img = numpy.array(img) / 127.5 - 1.
            img = img.reshape((1,)+target_size+(1,))
            img = vdsr.predict(img)
            print(filename)
            img = img.reshape(target_size+(1,))
            img = (0.5 * img + 0.5) * 255
            img = cv2.normalize(numpy.array(img), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            imsave('%s/%s' % (target_path, filename), img)
        else:
            pass

    # score(target_path)