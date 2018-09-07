from __future__ import print_function
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import TensorflowUtils as utils

from ImageGenerator import ImageDataGenerator
from ImageGenerator_fname import ImageDataGenerator_fname
#from tensorflow.contrib.data import Iterator

import datetime
from six.moves import xrange
import time
import os
import cv2

import multiprocessing
from tensorflow.python.client import device_lib

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("logs_dir", "logs_brain_FineTuning/", "path to logs directory")
tf.flags.DEFINE_float("learning_rate", "2e-5", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "visualize", "Mode train/ visualize")
tf.flags.DEFINE_string('optimization', "cross_entropy", "optimization mode: cross_entropy/ dice")

MAX_EPOCH = int(30)

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

if FLAGS.mode == 'train':
    is_training = True
    print("TRAINING...")
else:
    is_training = False
    print("NOT TRAINING...")

# Create "logs/" directory if not exists
if not os.path.exists(FLAGS.logs_dir):
    os.makedirs(FLAGS.logs_dir)
    print("Directory '" + FLAGS.logs_dir + "' created")

num_cpu = multiprocessing.cpu_count()
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

# NUM_OF_CLASSES = the number of segmentation classes + 1 (1 for none for anything)
NUM_OF_CLASSES = 2
IMAGE_SIZE = 224


def get_num_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])


def dice(mask1, mask2, smooth=1e-5):
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
    """,
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
            if FLAGS.debug:
                utils.add_activation_summary(current)
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
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
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
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    
    return optimizer.apply_gradients(grads)


def loss_and_grad(keep_probability, image, annotation, optimizer, batch_size=64, num_gpu=2):
    # Calculate the gradients and losses for each model tower.
    loss_tmp = []
    grad_tmp = []

    batch_per_gpu = int(batch_size / num_gpu)
    #keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")

    # Train all model
    # trainable_var = tf.trainable_variables()
    # ========================================
    # To limit the training range
    # scope_name = 'inference'
    # trainable_var = [var for var in trainable_var if scope_name in var.name]
    # ========================================

    with tf.variable_scope(tf.get_variable_scope()):
        for d in range(num_gpu):
            batch_start = batch_per_gpu * d
            batch_end = batch_per_gpu * (d + 1)
            image_part = image[batch_start:batch_end]
            annotation_part = annotation[batch_start:batch_end]
            
            with tf.device('/gpu:' + str(d)):
                # Calculate the loss for one tower of the model. This function
                # constructs the entire model but shares the variables across all towers.
                
                pred_annotation, logits = inference(image_part, keep_probability)

                if FLAGS.optimization == "cross_entropy":
                    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.squeeze(annotation_part, squeeze_dims=[3]),name="entropy"))  # For softmax
                
                elif FLAGS.optimization == "dice":
                    logits1 = tf.nn.softmax(logits) # default dim for softmax fn : -1 (in this case, 3, [N,224,224,2])
                    logits2 = tf.slice(logits1, [0,0,0,1],[-1,IMAGE_SIZE,IMAGE_SIZE,1]) # Remove the [0th] slice, where value 1 is 'no masks at all'
                    loss = 1 - tl.cost.dice_coe(logits2, tf.cast(annotation_part, dtype=tf.float32))

                # Reuse variables for the next tower.
                tf.get_variable_scope().reuse_variables()

                # Calculate the gradients for the batch of data on this tower.
                grad=optimizer.compute_gradients(loss)#, var_list=trainable_var)
                
                # Keep track of the gradients and loss across all towers.
                loss_tmp.append(loss)
                grad_tmp.append(grad)

    return loss_tmp, grad_tmp


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
       Note that this function provides a synchronization point across all towers.
       Args:
       tower_grads: List of lists of (gradient, variable) tuples. The outer list
                    is over individual gradients. The inner list is over the gradient
                    calculation for each tower.
       Returns:     List of pairs of (gradient, variable) where the gradient has been averaged
                    across all towers."""
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        # ( (grad0_gpu0, var0_gpu0), ..., (grad0_gpuN, var0_gpuN) )
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared across tower.
        # So .. we will just return the first tower's pointer to the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def main(argv=None):
    batch_size_per_gpu = 16
    num_gpu = get_num_gpus()
    batch_size = batch_size_per_gpu * num_gpu
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

    if is_training == True:
        tower_loss, tower_grad = loss_and_grad(keep_probability, image, annotation, optimizer, batch_size=batch_size, num_gpu=num_gpu)
        grad = average_gradients(tower_grad)
        train_op = optimizer.apply_gradients(grad)
    elif is_training == False:
        pred_annotation, logits = inference(image, keep_probability)
        if FLAGS.optimization == 'cross_entropy':
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.squeeze(annotation, squeeze_dims=[3]),name="entropy"))  # For softmax
        
        elif FLAGS.optimization == 'dice':
            logits1 = tf.nn.softmax(logits) # default dim for softmax fn : -1 (in this case, 3, [N,224,224,2])
            # Remove the 0 slice, where it masks "no masks"
            logits2 = tf.slice(logits1, [0,0,0,1],[-1,IMAGE_SIZE,IMAGE_SIZE,1])
            loss = 1 - tl.cost.dice_coe(logits2, tf.cast(annotation, dtype=tf.float32))


    """
    if FLAGS.optimization == "cross_entropy":
        annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")   # For cross entropy
        pred_annotation, logits = inference(image, keep_probability)
        print("pred_annotation, logits shape", pred_annotation.get_shape().as_list(), logits.get_shape().as_list())

        labels = tf.squeeze(annotation, squeeze_dims=[3])
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,name="entropy")) # For softmax

        #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.squeeze(annotation, squeeze_dims=[3]),name="entropy"))  # For softmax

    elif FLAGS.optimization == "dice":
        annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")  # For DICE
        pred_annotation, logits = inference(image, keep_probability)

        # pred_annotation (argmax) is not differentiable so it cannot be optimized. So in loss, we need to use logits instead of pred_annotation!
        logits1 = tf.nn.softmax(logits) # default dim for softmax fn : -1 (in this case, 3, [N,224,224,2])

        # Remove the 0 slice, where it masks "no masks"
        logits2 = tf.slice(logits1, [0,0,0,1],[-1,IMAGE_SIZE,IMAGE_SIZE,1])
        loss = 1 - tl.cost.dice_coe(logits2, tf.cast(annotation, dtype=tf.float32))
    """


    #total_var = tf.trainable_variables()
    # ========================================
    # To limit the training range
    # scope_name = 'inference'
    # trainable_var = [var for var in total_var if scope_name in var.name]
    # ========================================

    # Train all model
    #trainable_var = total_var
    #train_op = train(loss, trainable_var)

    #print("Setting up summary op...")
    #summary_op = tf.summary.merge_all()

    #for variable in trainable_var:
    #    print(variable)


    #Way to count the number of variables + print variable names
    """
    total_parameters = 0
    for variable in trainable_var:
        # shape is an array of tf.Dimension
        print(variable)
        shape = variable.get_shape()
        print(shape)
        print(len(shape))
        variable_parameters = 1
        for dim in shape:
            print(dim)
            variable_parameters *= dim.value
        print(variable_parameters)
        total_parameters += variable_parameters
    print("Total # of parameters : ", total_parameters)

    """
    # All the variables defined HERE -------------------------------
    dir_path = '/data'
    #mask_name = 'eyes/combine'
    mask_name = 'brain'

    range_min = -100
    range_max = 3000

    channel = 3

    opt_crop = False
    crop_shape = (224, 224)
    opt_resize = True
    resize_shape = (224, 224)
    rotation_status = True
    rotation_angle = [-5, 5]
    bitsampling = False
    bitsampling_bit = [4, 8]
    # --------------------------------------------------------------

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("Model restored...")
        saver.restore(sess, ckpt.model_checkpoint_path)

    if FLAGS.mode == "train":
        with tf.device('/cpu:0'):
            tr_data = ImageDataGenerator(dir_path=dir_path, mask_name=mask_name, img_resize=resize_shape, mask_resize=resize_shape, channel=channel, mode='training',
                                     rotation_status=rotation_status, rotation_angle=rotation_angle, batch_size=batch_size,num_threads=num_cpu, pix_upper_lim=range_max, pix_lower_lim=range_min)

            iterator = tf.data.Iterator.from_structure(tr_data.data.output_types, tr_data.data.output_shapes)
            next_batch = iterator.get_next()

            vd_data = ImageDataGenerator(dir_path=dir_path, mask_name=mask_name, img_resize=resize_shape, mask_resize=resize_shape, channel=channel, mode='validation',
                                     rotation_status=False, rotation_angle=rotation_angle, batch_size=batch_size,num_threads=num_cpu, pix_upper_lim=range_max, pix_lower_lim=range_min)

        train_batches_per_epoch = np.floor(tr_data.data_size / batch_size).astype(np.int16)
        validation_batches_per_epoch = np.floor(vd_data.data_size / batch_size).astype(np.int16)

        print("train batches per epoch : ", train_batches_per_epoch)
        print("valid batches per epoch : ", validation_batches_per_epoch)

        training_init_op = iterator.make_initializer(tr_data.data)
        validation_init_op = iterator.make_initializer(vd_data.data)
        
        print("Start training")
        start = time.time()
        train_loss_list = []
        x_train = []
        validation_loss_list = []
        x_validation = []
        
        for itr in xrange(MAX_EPOCH): 
            batch_itr = 0
            sess.run(training_init_op)

            while batch_itr < train_batches_per_epoch:
                train_images, train_annotations = sess.run(next_batch)
                feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.75} # 0.75 before
                sess.run(train_op, feed_dict=feed_dict)
                
                end = time.time()
                print("Epoch # ", itr+1, ", Batch # ", batch_itr+1 , ", ", np.int32(end - start), "s")
                batch_itr = batch_itr + 1

            train_loss = sess.run(tower_loss, feed_dict={image: train_images, annotation: train_annotations, keep_probability: 1.0})
            train_loss = np.mean(train_loss)
            print("EPOCH: %d, Train_loss:%g" % (itr+1, train_loss))
            train_loss_list.append(train_loss)
            x_train.append(itr+1)
            

            # Validation loss calculation 
            batch_val_itr = 0
            temp_valid_loss = []
            
            sess.run(validation_init_op)

            while batch_val_itr < validation_batches_per_epoch:
                print(batch_val_itr, ", ", validation_batches_per_epoch)
                valid_images, valid_annotations = sess.run(next_batch) 
                temp_loss = sess.run(tower_loss, feed_dict={image: valid_images, annotation: valid_annotations, keep_probability: 1.0})
                temp_valid_loss.append(np.mean(temp_loss))
                batch_val_itr = batch_val_itr + 1


            valid_loss = np.mean(temp_valid_loss)
            print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
            validation_loss_list.append(valid_loss)
            x_validation.append(itr+1)

            end = time.time()
            print("Epoch #", itr+1, ",", np.int32(end - start), "s")

            # Save it every 30 epoch
            if (itr+1)%50 == 0:
            	fname = mask_name.replace("/","_") + "_" + str(range_min) + "_" + str(range_max) + "_FCN_" + FLAGS.optimization
            	saver.save(sess, FLAGS.logs_dir + fname + ".ckpt", itr+1)

        
        fname = mask_name.replace("/","_") + "_" + str(range_min) + "_" + str(range_max) + "_FCN_" + FLAGS.optimization
        saver.save(sess, FLAGS.logs_dir + fname + ".ckpt", itr+1)

        # Draw loss functions
        plt.plot(x_train,train_loss_list,label='train')
        plt.plot(x_validation,validation_loss_list,label='validation')
        plt.title("loss functions")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.ylim(ymin=min(train_loss_list))
        plt.ylim(ymax=max(train_loss_list)*1.1)
        plt.legend()
        plt.savefig(FLAGS.logs_dir + "/loss_functions.png")
        #plt.show()

        print("train loss list : ", train_loss_list)
        print("valid loss list : ", validation_loss_list)


    elif FLAGS.mode == "visualize":
        with tf.device('/cpu:0'):
            vd_data = ImageDataGenerator_fname(dir_path=dir_path, mask_name=mask_name, img_resize=resize_shape, mask_resize=resize_shape, channel=channel, mode='validation',
                                               rotation_status=False, rotation_angle=rotation_angle, batch_size=1,num_threads=num_cpu, pix_upper_lim=range_max, pix_lower_lim=range_min)

            iterator = tf.data.Iterator.from_structure(vd_data.data.output_types, vd_data.data.output_shapes)
            next_batch = iterator.get_next()
    
        validation_init_op = iterator.make_initializer(vd_data.data)
        sess.run(validation_init_op)

        dice_array = []
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        # Save the image for display. Use matplotlib to draw this.
        for itr in range(84):
            valid_images, valid_annotations, img_fname = sess.run(next_batch)
            img_fname = img_fname[0].decode('utf-8')
            img_fname = img_fname.split('/')[-1] 
            img_fname = img_fname.split('.')[0]
            print(img_fname)

            print("type : ", valid_images.shape, valid_images.dtype, valid_annotations.shape, valid_annotations.dtype)
            
            if FLAGS.optimization == 'cross_entropy':
                pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations, keep_probability: 1.0})
                print("cross_entropy!")
            elif FLAGS.optimization == 'dice':
                pred = sess.run(logits2, feed_dict={image: valid_images, annotation: valid_annotations, keep_probability: 1.0})
                print("dice!")
            
            pred = np.squeeze(pred, axis=3)

            valid_annotations = np.squeeze(valid_annotations, axis=3)
            print("pred, annotations : ", pred.shape, valid_annotations.shape)
            dice_coeff = dice(valid_annotations[0], pred[0])

            dice_array.append(dice_coeff)
            print("min max of prediction : ", pred.flatten().min(), pred.flatten().max())
            print("min max of validation : ", valid_annotations.flatten().min(), valid_annotations.flatten().max())
            print("DICE : ", dice_coeff)

            # Save images
            print(valid_images[0].shape, valid_annotations[0].shape, pred[0].shape)
            print(valid_images.dtype, valid_annotations.dtype, pred.dtype)
            fname = FLAGS.logs_dir + "/Prediction_" + str(itr) + "_" + str(dice_coeff) + "_" + img_fname + ".png"
            mask_color_img(valid_images[0], valid_annotations[0], pred[0], dice_coeff, fname, shape=valid_images[0].shape, alpha=0.4)

        plt.figure()
        plt.hist(dice_array,bins)
        plt.xlabel('Dice')
        plt.ylabel('frequency')
        plt.title('Dice coefficient distribution of validation dataset')
        plt.savefig(FLAGS.logs_dir + "/dice_histogram" + ".png")
        plt.close()
        #plt.show()



def mask_color_img(img, mask, pred, dice, fname, shape=(224,224,3), alpha=0.6):
    color_mask = np.zeros(shape=shape, dtype=np.float32)
    color_mask[:,:,0] = mask # 0 -> red, 1 -> green, 2 -> blue
    
    color_pred = np.zeros(shape=shape, dtype=np.float32)
    color_pred[:,:,0] = pred # 0 -> red, 1 -> green, 2 -> blue

    img = img / np.max(img)

    img_mask = cv2.addWeighted(color_mask, alpha, img, 1-alpha ,0)
    img_pred = cv2.addWeighted(color_pred, alpha, img, 1-alpha ,0)

    f, (ax0, ax1) = plt.subplots(1, 2, subplot_kw={'xticks': [], 'yticks': []})

    ax0.imshow(img_mask)
    ax0.set_title("Original")
    ax1.imshow(img_pred)
    ax1.set_title("Prediction")

    plt.suptitle("DICE : " + str(dice))
    plt.savefig(fname)
    plt.close()


if __name__ == "__main__":
    tf.app.run()
