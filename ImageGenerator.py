# Created on Wed May 31 14:48:46 2017
# @author: Frederik Kratzert
# @modified by Dong Joo Rhee

"""Containes a helper class for image input pipelines in tensorflow."""

import tensorflow as tf
import numpy as np
import glob
import dicom_utils.dicomPreprocess as dpp
import skimage.transform as sktf


class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.
    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, dir_path, mask_name, img_resize, mask_resize, channel, mode, rotation_status, rotation_angle, batch_size, shuffle=True, buffer_size=5, num_threads=10, 
                 pix_upper_lim=2000,pix_lower_lim=-1000):
        """Create a new ImageDataGenerator.
        Recieves a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and seperated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensrFlow datasets, that can be used to train
        e.g. a convolutional neural network.
        Args:
            dir_path: Path to the files.
            mode: Either 'training' or 'validation'. Depending on this value,
                different parsing functions will be used.
            batch_size: Number of images per batch.
            shuffle: Whether or not to shuffle the data in the dataset and the
                initial file list.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.
        Raises:
            ValueError: If an invalid mode is passed.
        """

        self.img_path = dir_path + '/image'

        if mode == 'training':
            self.mask_path = dir_path + '/' + mask_name + '/training'
        elif mode == 'validation':
            self.mask_path = dir_path + '/' + mask_name + '/validation' 
        else:
            print("Warning ---- Wrong mode!!")
            return
       
        print(self.mask_path)

        self.img_resize = img_resize
        self.mask_resize = mask_resize
        self.channel = channel
        self.mask_list = []
        self.img_list = []
        self.rotation_status = rotation_status
        self.rotation_angle = rotation_angle
        self.mode = mode

        # TODO 1: Check what buffer size is.
        with open(self.mask_path + '/image_slice_index.dat','r') as img_file, open(self.mask_path + '/mask_slice_index.dat','r') as mask_file:
            self.img_list = img_file.read().splitlines()
            self.mask_list = mask_file.read().splitlines()

        self.img_list = [self.img_path + '/' + x for x in self.img_list]
        self.mask_list = [self.mask_path + '/' + x for x in self.mask_list]

        img_file.close()
        mask_file.close()

        # number of samples in the dataset
        self.data_size = len(self.mask_list)

        # initial shuffling of the image and mask lists (together!)
        if shuffle:
            self._shuffle_lists()

        print("-"*50)
        print("First img, mask names : ", self.img_list[0], self.mask_list[0])
        print("-"*50)

        # create dataset
        data = tf.data.Dataset.from_tensor_slices((self.img_list, self.mask_list))

        data = data.map(lambda img_fname, mask_fname : tuple(tf.py_func(self._read_py_function, [img_fname, mask_fname, pix_upper_lim, pix_lower_lim],
                                                                        [tf.double, tf.float64])), num_parallel_calls=num_threads)

        # Set image/mask types to be uint16, bool respectively. Then, convert them into
        # float32/float32 (? think about it) for feeding

        # TODO 2: shuffle everytime repeated... shuffle must come after repeat
        #data = data.repeat()

        data = data.shuffle(buffer_size=int(self.data_size/4)) # MAKE THIS ONE FAST!!!!
        data = data.batch(batch_size)

        if mode == 'training' or mode == 'validation':
            data = data.map(self.channel_function, num_parallel_calls=num_threads)
            #  Currently have some problem + not compatible with what I try to do..
            #data = data.apply(tf.contrib.data.map_and_batch(map_func=self._parse_function_train, batch_size=batch_size))
        elif mode == 'test':
            data = data.map(self.test_function, num_parallel_calls=num_threads)
        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        data = data.prefetch(buffer_size=buffer_size)

        self.data = data


    def _read_py_function(self, img_fname, mask_fname, pix_upper_lim, pix_lower_lim):
        img_fname = img_fname.decode('utf-8')
        mask_fname = mask_fname.decode('utf-8')

        image = np.load(img_fname) #dtype = uint16
        image = dpp.resize_2d(image, self.img_resize)
        mask = np.load(mask_fname) #dtype = uint16
        mask = dpp.resize_2d(mask, self.mask_resize)

        if self.rotation_status == True and self.mode == 'training':
            aug_img = image[np.newaxis, :, :]
            aug_mask = mask[np.newaxis, :, :]
            for i in range(len(self.rotation_angle)):
                rotate_img = dpp.rotate(image, self.rotation_angle[i])
                rotate_mask = dpp.rotate(mask, self.rotation_angle[i])
                aug_img = np.append(aug_img, rotate_img[np.newaxis, :, :], axis=0)
                aug_mask = np.append(aug_mask, rotate_mask[np.newaxis, :, :], axis=0)
        else:
            aug_img = image
            aug_mask = mask

        # Set the range of pixel values to be 0 - 3000
        aug_img[aug_img < pix_lower_lim] = pix_lower_lim # Value less than pixel lower limit -> pixel lower limit
        aug_img[aug_img > pix_upper_lim] = pix_upper_lim   # Value more than 2000 -> pixel upper limit
        
        aug_img = aug_img - np.min(aug_img.flatten()) # range from (lower_lim, upper_lim) -> (0,upper_lim - lower_lim)

        # Normalize pixel values to (0,255) range
        pix_range = pix_upper_lim - pix_lower_lim
        bin_size = pix_range/255
        aug_img = np.matrix.round(aug_img / bin_size) # range from (0,upper_lim - lower_lim) -> (0,255)
        
        aug_mask = np.round(aug_mask) # Mask to be int later, so round the value

        return aug_img, aug_mask


    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths."""
        img_path = self.img_list
        mask_path = self.mask_list
        permutation = np.random.permutation(self.data_size)
        self.img_list = []
        self.mask_list = []
        for i in permutation:
            self.img_list.append(img_path[i])
            self.mask_list.append(mask_path[i])


    def channel_function(self, img, mask):
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

        mask = tf.reshape(mask, shape=(-1, self.mask_resize[0], self.mask_resize[1]))
        mask = tf.expand_dims(mask, -1)

        mask = tf.cast(mask, tf.int32)
        img = tf.cast(img, tf.float32)

        return img, mask

    def test_function(self, img, mask):

        """
        #Data augmentation + 3 channel addition comes here.
        """
        img = tf.reshape(img, shape=(-1, self.img_resize[0], self.img_resize[1]))
        img = tf.expand_dims(img, -1)

        return img
