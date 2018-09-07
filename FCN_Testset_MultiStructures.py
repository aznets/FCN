from __future__ import print_function
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import TensorflowUtils as utils

from ImageGenerator_testset import ImageDataGenerator_testset

import datetime
from six.moves import xrange
import time
import os, sys, glob
import cv2

from dicom_utils import color as cl

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs_total/", "path to logs directory")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_string('optimization', "dice", "optimization mode: cross_entropy/ dice")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

os.environ["CUDA_VISIBLE_DEVICES"]="3"

# NUM_OF_CLASSES = the number of segmentation classes + 1 (1 for none for anything)
NUM_OF_CLASSES = 2
IMAGE_SIZE = 224

def dice(mask1, mask2, smooth=1e-6):
    print("-"*50)
    print("mask, pred shape : ", mask1.shape, mask2.shape)
    print("max, min mask : ", mask1.flatten().max(), mask2.flatten().max())
    print("max, min pred : ", np.sum(mask1.flatten()), np.sum(mask2.flatten()))

    mul = mask1 * mask2
    inse = np.sum(mul.flatten())

    l = np.sum(mask1.flatten())
    r = np.sum(mask2.flatten())

    dice_coeff = (2.* inse + smooth) / (l + r + smooth)

    return round(dice_coeff,3)

def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3'
    )
    """
        'relu5_3', 'conv5_4', 'relu5_4'
    )"""

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w",)
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net


def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]

        pool5 = utils.max_pool_2x2(conv_final_layer)

    with tf.variable_scope("FCN"):
        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")

        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSES], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSES], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSES], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSES])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSES, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSES], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

        print("anno, conv_t3 shape", tf.shape(annotation_pred), tf.shape(conv_t3))

    return tf.expand_dims(annotation_pred, dim=3), conv_t3


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(1)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    
    return optimizer.apply_gradients(grads)


def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")

    if FLAGS.optimization == "cross_entropy":
        annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")   # For cross entropy
        pred_annotation, logits = inference(image, keep_probability)
        print("pred_annotation, logits shape", pred_annotation.get_shape().as_list(), logits.get_shape().as_list())

        label = tf.squeeze(annotation, squeeze_dims=[3])
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label,name="entropy")) # For softmax

        #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.squeeze(annotation, squeeze_dims=[3]),name="entropy"))  # For softmax

    elif FLAGS.optimization == "dice":
        annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")  # For DICE
        pred_annotation, logits = inference(image, keep_probability)

        # pred_annotation (argmax) is not differentiable so it cannot be optimized. So in loss, we need to use logits instead of pred_annotation!
        logits1 = tf.nn.softmax(logits) # default dim for softmax fn : -1 (in this case, 3, [N,224,224,2])

        # Remove the 0 slice, where it masks "no masks"
        logits2 = tf.slice(logits1, [0,0,0,1],[-1,IMAGE_SIZE,IMAGE_SIZE,1])
        loss = 1 - tl.cost.dice_coe(logits2, tf.cast(annotation, dtype=tf.float32))



    total_var = tf.trainable_variables()

    # Train all model
    trainable_var = total_var
    train_op = train(loss, trainable_var)

    # All the variables defined HERE -------------------------------
    dir_path = '/data/HN_image_only'

    batch_size = 32
    channel=3

    opt_crop = False
    crop_shape = (224, 224)
    opt_resize = True
    resize_shape = (224, 224)
    
    organ_list = ['Brain', 'Esophagus','Mandible','Parotid','cord', 'brainstem']
    # --------------------------------------------------------------

    organ_name = []
    pred_list = []
    ckpt_list = []
    
    # Matching organ with corresponding ckpt file
    names = [os.path.basename(x) for x in glob.glob(FLAGS.logs_dir + "*.ckpt*.data*")]
    for i, oname in enumerate(organ_list):
        for fname in names:
            if oname.lower() == fname.split('_')[0].lower():
                ckpt_list.append('.'.join(fname.split('.')[0:2]))

    print(organ_list)
    print(ckpt_list)

    # If not all ckpt exist for given organ_list
    if len(ckpt_list) != len(organ_list):
        print("Don't have enough ckpt files!")
        sys.exit()


    # Save the image for display. 
    # Number of patients here
    for itr in range(3):
        pred_list = []
        test_images = 0
        organ_name = []

        with tf.Session() as sess:
            for i, organ in enumerate(organ_list):
                organ_name.append(organ)
                temp = ckpt_list[i].split('.')[0]
                temp = temp.split('_')[1:3]
                pix_lower_lim = int(temp[0])
                pix_upper_lim = int(temp[1])
                print(pix_lower_lim, pix_upper_lim)
            
                # 1. Call patient image data for each organ due to pixel intensity variation
                with tf.device('/cpu:0'):
                    test_data = ImageDataGenerator_testset(dir_path=dir_path, img_resize=resize_shape, channel=channel, patient_num=itr, pix_upper_lim=pix_upper_lim, pix_lower_lim=pix_lower_lim)
                    iterator = tf.data.Iterator.from_structure(test_data.data.output_types, test_data.data.output_shapes)
                    next_batch = iterator.get_next()
    
                    test_init_op = iterator.make_initializer(test_data.data)
                    sess.run(test_init_op)

                test_images = sess.run(next_batch)
                zdim = test_images.shape[0]

                # 2. Create mask for each organ
                print("Setting up Saver...")
                saver = tf.train.Saver()
        
                sess.run(tf.global_variables_initializer())
                print("Model restored...")
                saver.restore(sess, FLAGS.logs_dir + ckpt_list[i])

                if FLAGS.optimization == 'cross_entropy':
                    pred = sess.run(pred_annotation, feed_dict={image: test_images, keep_probability: 1.0})
                    print("cross_entropy!")
                elif FLAGS.optimization == 'dice':
                    pred = sess.run(logits2, feed_dict={image: test_images, keep_probability: 1.0})
                    print("dice!")
                
                pred = np.squeeze(pred, axis=3)
                pred_list.append(pred)

                # 3. Draw the masks on the image
                print(organ_name)
                print(len(pred_list), pred_list[0].shape)
                print(test_images.shape)

        mask_color_img(test_images, pred_list, color_list=organ_name, itr=itr, alpha=0.6)


def mask_color_img(img, pred, color_list, itr, alpha=0.6):
    print("Save Images in png file")
    print(img.shape[0])
    for slice_num in range(img.shape[0]):
        img_slice = np.squeeze(img[slice_num])
        img_slice = img_slice / np.max(img_slice)
        
        color_pred = np.zeros(shape=img_slice.shape, dtype=np.float32)

        for organ_num, organ_pred in enumerate(pred):
            pred_slice = organ_pred[slice_num]
            pred_slice[pred_slice >= 0.5] = 1
            pred_slice[pred_slice < 0.5] = 0
            color = cl.HN_organ_color(color_list[organ_num])
        
            color_pred[:,:,0] = color_pred[:,:,0] + [x*color[0] for x in pred_slice] # 0 -> red, 1 -> green, 2 -> blue
            color_pred[:,:,1] = color_pred[:,:,1] + [x*color[1] for x in pred_slice] # need to modify this part, as it does not describe black...
            color_pred[:,:,2] = color_pred[:,:,2] + [x*color[2] for x in pred_slice]

        img_pred = cv2.addWeighted(color_pred, alpha, img_slice, 1-alpha ,0)

        f, ax0 = plt.subplots(1, 1, subplot_kw={'xticks': [], 'yticks': []})

        ax0.imshow(img_pred)
        ax0.set_title("Prediction")

        plt.savefig(FLAGS.logs_dir + "/Patient" + str(itr) + "_" + str(slice_num) + ".png")
        plt.close()



if __name__ == "__main__":
    tf.app.run()
