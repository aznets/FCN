# Created on Wed May 31 14:48:46 2017
# @author: Frederik Kratzert
# @modified by Dong Joo Rhee

"""Containes a helper class for image input pipelines in tensorflow."""

import tensorflow as tf
import numpy as np
import glob
import dicom_utils.dicomPreprocess as dpp
import skimage.transform as sktf
import scipy.io as sio
import glob, os

class ImageDataGenerator_testset(object):
    """Wrapper class around the new Tensorflows dataset pipeline.
    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, dir_path, img_resize, channel, patient_num=0, buffer_size=1, num_threads=1, 
                 pix_upper_lim=3000,pix_lower_lim=-1000):
        """Create a new ImageDataGenerator.
        Recieves a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and seperated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensrFlow datasets, that can be used to train
        e.g. a convolutional neural network.
        Args:
            dir_path: Path to the files.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.
        Raises:
            ValueError: If an invalid mode is passed.
        """

        self.img_path = dir_path 
       
        self.img_resize = img_resize
        self.channel = channel
        self.mask_list = []
        self.img_list = []

        for file in glob.glob(self.img_path+"/*.mat"):
            self.img_list.append(file)

        test = []
        test.append(self.img_list[patient_num])

        print(test)
        # create dataset
        #data = tf.data.Dataset.from_tensor_slices((self.img_list))
        data = tf.data.Dataset.from_tensor_slices((test))
        data = data.map(lambda img_fname : tuple(tf.py_func(self._read_py_function, [img_fname, pix_upper_lim, pix_lower_lim], [tf.double])), num_parallel_calls=num_threads)

        data = data.map(self.channel_function, num_parallel_calls=num_threads)
        #data = data.prefetch(buffer_size=buffer_size)
        self.data = data


    def _read_py_function(self, img_fname, pix_upper_lim, pix_lower_lim):
        img_fname = img_fname.decode('utf-8')

        # read matlab file and turn it into numpy
        print("img_fname : ", img_fname)
        a = sio.loadmat(img_fname)
        image = a['img']

        image = dpp.resize_3d(image, self.img_resize)

        # Set the range of pixel values to be 0 - 3000
        image[image < pix_lower_lim] = pix_lower_lim # Value less than pixel lower limit -> pixel lower limit
        image[image > pix_upper_lim] = pix_upper_lim   # Value more than 2000 -> pixel upper limit
        
        image = image - np.min(image.flatten()) # range from (lower_lim, upper_lim) -> (0,upper_lim - lower_lim)

        # Normalize pixel values to (0,255) range
        pix_range = pix_upper_lim - pix_lower_lim
        bin_size = pix_range/255
        image = np.matrix.round(image / bin_size) # range from (0,upper_lim - lower_lim) -> (0,255)
        
        return image


    def channel_function(self, img):
        """
        3 channel addition comes here.
        """
        img = tf.reshape(img, shape=(-1, self.img_resize[0], self.img_resize[1]))
        if self.channel == 3:
            img = tf.stack([img, img, img], axis=3)
        elif self.channel == 1:
            img = tf.expand_dims(img, -1)
        else:
            print("Number of channel is neither 1 nor 3, but ", self.channel)
            img = tf.stack([img, img, img], axis=3)

        img = tf.cast(img, tf.float32)

        return img
