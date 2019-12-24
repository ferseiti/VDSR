#!/usr/bin/env python3

from __future__ import print_function
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Lambda, BatchNormalization
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D, Input, ZeroPadding2D, merge, add
import tensorflow as tf
from keras.models import load_model
from keras import optimizers
from keras import losses
from keras.optimizers import SGD, Adam, Adagrad, Adadelta
from keras.callbacks import ModelCheckpoint, Callback, callbacks
from keras.preprocessing.image import ImageDataGenerator

import os, glob, sys, threading
import scipy.io
import h5py
from PIL import Image

from tifffile import imread
import matplotlib.pyplot as plt

import time
import numpy
import re
import math
import argparse

from keras.backend.tensorflow_backend import set_session

import signal
import time

def readConfiguration(signalNumber, frame):
    print ('(SIGHUP) reading configuration')
    return

def terminateProcess(signalNumber, frame):
    print ('(SIGTERM) terminating the process')
    sys.exit()

def receiveSignal(signalNumber, frame):
    print('Received:', signalNumber)
    return

DATA_PATH = "data/train_data.h5"
LABEL_PATH = "data/label_data.h5"
VAL_PATH = "data/validation.h5"
VAL_LABEL_PATH = "data/validation_label.h5"
TRAIN_SCALES = [2]
same_SCALES = [2]

class MyCallback(Callback):
    
    def __init__(self, model):
         self.model_to_save = model

    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print('Learning rate with decay: {}'.format(K.eval(lr_with_decay)))
        print('Decay: {}'.format(K.eval(decay)))
        print('Initial learning rate: {}'.format(K.eval(lr)))

def tf_log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def load_images(path):
    with h5py.File(path, 'r') as fd:
        images = numpy.array(fd['data'])
        images = ((images - images.min()) * 1.000 / (images.max() - images.min()))
    return images

def get_image_list(data_path, scales=[2, 3, 4]):
    l = glob.glob(os.path.join(data_path,"*"))
    print(len(l))
    l = [f for f in l if re.search("^\d+.tif$", os.path.basename(f))]
    print(len(l))
    
    train_list = []
    for f in l:
        if os.path.exists(f):
            for i in range(len(scales)):
                scale = scales[i]
                string_scale = "_" + str(scale) + ".tif"
                if os.path.exists(f[:-4]+string_scale): train_list.append([f, f[:-4]+string_scale])
    print(l)
    return train_list

def get_image_batch(h5_file_path, offset, batch_size):

    print('Reading file {}, offset {}'.format(h5_file_path, offset), end='\r')
    sys.stdout.write("\033[K") #clear line
    with h5py.File(h5_file_path) as h5fd:
        shape = h5fd['data'].shape
        data = numpy.array(h5fd['data'][offset:offset+batch_size])
        label = numpy.array(h5fd['label'][offset:offset+batch_size])

    return data, label, shape

def resize_data(train_data):
    
    batch_x = []
    for t in train_data:
        x = numpy.zeros(img_size)
        #x[:,:,0] = misc.imresize(t[:,:,0], size=(img_size[0],img_size[1]), interp='bicubic')
        x[:,:,0] = numpy.array(Image.fromarray(t[:,:,0]).resize((img_size[0],img_size[1]), Image.BICUBIC))
        batch_x.append(x)
    batch_x = numpy.array(batch_x)
    
    return batch_x

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def image_gen(target_list, batch_size):
    offset = 0
    target_count = 0
    while True:
        for target in target_list:
            batch_x, batch_y, shape = get_image_batch(target, offset, batch_size)
            if target_count == len(target_list):
                offset += batch_size
                target_count = 0
            if offset >= shape[0]:
                offset = 0
            target_count += 1
            yield (batch_x, batch_y)

def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true)))) 

def PSNR_loss(y_true, y_pred):
    max_pixel = 1.0
    return -10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true)))) 

# SSIM loss function
def ssim(y_true, y_pred):
  return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))

def ssim_metric(y_true, y_pred):
    # source: https://gist.github.com/Dref360/a48feaecfdb9e0609c6a02590fd1f91b

    y_true = tf.transpose(y_true, [0, 2, 3, 1])
    y_pred = tf.transpose(y_pred, [0, 2, 3, 1])
    patches_true = tf.extract_image_patches(y_true, [1, 5, 5, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")
    patches_pred = tf.extract_image_patches(y_pred, [1, 5, 5, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")

    u_true = K.mean(patches_true, axis=3)
    u_pred = K.mean(patches_pred, axis=3)
    var_true = K.var(patches_true, axis=3)
    var_pred = K.var(patches_pred, axis=3)
    std_true = K.sqrt(var_true)
    std_pred = K.sqrt(var_pred)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    ssim /= denom
    ssim = tf.where(tf.is_nan(ssim), K.zeros_like(ssim), ssim)
    return ssim


def crop(dimension, start, end):
    ## from https://github.com/keras-team/keras/issues/890#issuecomment-319671916
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)

# Get the training and testing data
# train_list = get_image_list("./data/x/", scales=TRAIN_SCALES)

# test_list = get_image_list("./data/val/", scales=same_SCALES)

def model_train(img_size, batch_size, epochs, optimizer, learning_rate, train_list, validation_list, style=2):

    print('Style {}.'.format(style))

    if style == 1:
        input_img = Input(shape=img_size)

        #model = Sequential()

        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', input_shape=img_size)(input_img)
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = Activation('relu')(model)

        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = Activation('relu')(model)

        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = Activation('relu')(model)

        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = Activation('relu')(model)
        model = Conv2D(1, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        res_img = model

        output_img = merge.Add()([res_img, input_img])

        model = Model(input_img, output_img)

        #model.load_weights('vdsr_model_edges.h5')

        adam = Adam(lr=0.000005)
        #sgd = SGD(lr=1e-3, momentum=0.9, decay=1e-4, nesterov=False)
        sgd = SGD(lr=0.01, momentum=0.9, decay=0.001, nesterov=False)
        #model.compile(sgd, loss='mse', metrics=[PSNR, "accuracy"])
        model.compile(adam, loss='mse', metrics=[ssim, ssim_metric, PSNR, "accuracy"])

        model.summary()

    else:

        input_img = Input(shape=img_size)

        model = Conv2D(64, (3, 3), padding='valid', kernel_initializer='he_normal', use_bias=False)(input_img)
        model = BatchNormalization()(model)
        model_0 = Activation('relu')(model)

        total_conv = 22  # should be even number
        total_conv -= 2  # subtract first and last
        residual_block_num = 5  # should be even number

        for _ in range(residual_block_num):  # residual block
            model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(model_0)
            model = BatchNormalization()(model)
            model = Activation('relu')(model)
            for _ in range(int(total_conv/residual_block_num)-1):
                model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
                model = Activation('relu')(model)
                model_0 = add([model, model_0])

        model = Conv2DTranspose(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = Conv2D(1, (3, 3), padding='valid', kernel_initializer='he_normal')(model)
        
        res_img = model

        input_img1 = crop(1,2,-2)(input_img)
        input_img1 = crop(2,2,-2)(input_img1)

        print(input_img.shape)
        print(input_img1.shape)
        output_img = merge.Add()([res_img, input_img1])
        # output_img = res_img
        model = Model(input_img, output_img)

        # model.load_weights('./vdsr_model_edges.h5')
        # adam = Adam(lr=learning_rate)
        adam = Adadelta()
        # sgd = SGD(lr=1e-7, momentum=0.9, decay=1e-2, nesterov=False)
        sgd = SGD(lr=learning_rate, momentum=0.9, decay=1e-4, nesterov=False, clipnorm=1)
        if optimizer == 0:
            model.compile(adam, loss='mse', metrics=[ssim, ssim_metric, PSNR])
        else:
            model.compile(sgd, loss='mse', metrics=[ssim, ssim_metric, PSNR])


        model.summary()

    mycallback = MyCallback(model)
    timestamp = time.strftime("%m%d-%H%M", time.localtime(time.time()))
    csv_logger = callbacks.CSVLogger('data/callbacks/training_{}.log'.format(timestamp))
    filepath="./checkpoints/weights-improvement-{epoch:03d}-{PSNR:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor=PSNR, verbose=1, mode='max')
    callbacks_list = [mycallback, checkpoint, csv_logger]

    with open('./model/vdsr_architecture.json', 'w') as f:
        f.write(model.to_json())

    history = model.fit_generator(image_gen(train_list, batch_size=batch_size), 
                        #steps_per_epoch=384400*len(train_list) // batch_size,
                        steps_per_epoch=384400// batch_size,
                        # steps_per_epoch=4612800//batch_size,
                        validation_data=image_gen(validation_list,batch_size=batch_size),
                        validation_steps=384400 // (batch_size*5),
                        #validation_steps=384400*len(validation_list) // batch_size,
                        epochs=epochs,
                        workers=1024,
                        callbacks=callbacks_list,
                        verbose=1)

    print("Done training!!!")

    print("Saving the final model ...")

    model.save('vdsr_model.h5')  # creates a HDF5 file 
    del model  # deletes the existing model


    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'validation'], loc='upper left')
    # plt.show()
    plt.savefig('loss.png')

    plt.plot(history.history['PSNR'])
    plt.plot(history.history['val_PSNR'])
    plt.title('Model PSNR')
    plt.ylabel('PSNR')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'validation'], loc='upper left')
    # plt.show()
    plt.savefig('PSNR.png')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Train a VDSR.')
    parser.add_argument('-s', '--size', help='Size of one dimension of the square image.', required=True, type=int)
    parser.add_argument('-b', '--batch-size', help='Batch size.', required=True, type=int, default=64)
    parser.add_argument('-e', '--epochs', help='Amount of epochs.', required=True, type=int, default=100)
    parser.add_argument('-o', '--optimizer', help='0: Adam, 1: SGD.', required=True, type=int, choices={0, 1}, default=0)
    parser.add_argument('-l', '--lr', help='Learning rate value.', required=True, type=float, default=0.00001)
    parser.add_argument('-t', '--train-list', help='List of H5 files paths where the training data is located.', required=True)
    parser.add_argument('-v', '--validation-list', help='List of H5 files paths where the validation data is located.', required=True)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    set_session(tf.Session(config=config))

    arguments = parser.parse_args()

    try:
        with open(arguments.train_list) as tfd:
            train_list = []
            for l in tfd.readlines():
                train_list.append(l.strip())
    except Exception as e:
        print('Insame training file list', file=sys.stderr)
        sys.exit(e)
    
    try:
        with open(arguments.validation_list) as vfd:
            validation_list = []
            for l in vfd.readlines():
                validation_list.append(l.strip())
    except Exception as e:
        print('Insame validation file list', file=sys.stderr)
        sys.exit(e)

    batch_size = arguments.batch_size
    size = (arguments.size, arguments.size, 1)
    epochs = arguments.epochs
    optimizer = arguments.optimizer
    learning_rate = arguments.lr

    model_train(size, batch_size, epochs, optimizer, learning_rate, train_list, validation_list, 2)
