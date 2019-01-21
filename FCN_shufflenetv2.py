from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim

import numpy as np
import os
from model import model_fn, RestoreMovingAverageHook



import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "4", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/FS181025_9/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/data/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for SGD Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/shufflenetv2_imagenet/", "Path to ShuffleNetV2 model")
tf.flags.DEFINE_string("pupil_seg_model_dir", "Model_zoo/shufflenetv2_imagenet_pupil_seg/", "Path to ShuffleNetV2 model")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

# MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(200000 + 1)
NUM_OF_CLASSESS = 2
IMAGE_SIZE = 512
IMAGE_SIZE_W = 640
IMAGE_SIZE_H = 480

BATCH_NORM_MOMENTUM = 0.997
BATCH_NORM_EPSILON = 1e-3

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.1
end_learning_rate = 0.0000000001
decay_steps = 200000

decay_learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step,
                                          decay_steps, end_learning_rate,
                                          power=0.5)


def shufflenet(images, is_training, num_classes=1000, depth_multiplier='1.0'):
    """
    This is an implementation of ShuffleNet v2:
    https://arxiv.org/abs/1807.11164

    Arguments:
        images: a float tensor with shape [batch_size, image_height, image_width, 3],
            a batch of RGB images with pixel values in the range [0, 1].
        is_training: a boolean.
        num_classes: an integer.
        depth_multiplier: a string, possible values are '0.5', '1.0', '1.5', and '2.0'.
    Returns:
        a float tensor with shape [batch_size, num_classes].
    """
    possibilities = {'0.5': 48, '1.0': 116, '1.5': 176, '2.0': 224}
    initial_depth = possibilities[depth_multiplier]

    def batch_norm(x):
        x = tf.layers.batch_normalization(
            x, axis=3, center=True, scale=True,
            training=is_training,
            momentum=BATCH_NORM_MOMENTUM,
            epsilon=BATCH_NORM_EPSILON,
            fused=True, name='batch_norm'
        )
        return x

    with tf.name_scope('standardize_input'):
        x = (2.0 * images) - 1.0

    with tf.variable_scope('ShuffleNetV2'):
        params = {
            'padding': 'SAME', 'activation_fn': tf.nn.relu,
            'normalizer_fn': batch_norm, 'data_format': 'NHWC',
            'weights_initializer': tf.contrib.layers.xavier_initializer()
        }
        with slim.arg_scope([slim.conv2d, depthwise_conv], **params):

            x = slim.conv2d(x, 24, (3, 3), stride=2, scope='Conv1')
            x = slim.max_pool2d(x, (3, 3), stride=2, padding='SAME', scope='MaxPool')

            x = block(x, num_units=4, out_channels=initial_depth, scope='Stage2')
            x1 = x

            x = block(x, num_units=8, scope='Stage3')
            x2 = x

            x = block(x, num_units=4, scope='Stage4')
            # x3 = x
            final_channels = 1024 if depth_multiplier != '2.0' else 2048
            x = slim.conv2d(x, final_channels, (1, 1), stride=1, scope='Conv5')
            x3 = x

    # global average pooling
    x = tf.reduce_mean(x, axis=[1, 2])

    logits = slim.fully_connected(
        x, num_classes, activation_fn=None, scope='classifier',
        weights_initializer=tf.contrib.layers.xavier_initializer()
    )
    return x1, x2, x3


def block(x, num_units, out_channels=None, scope='stage'):
    with tf.variable_scope(scope):

        with tf.variable_scope('unit_1'):
            x, y = basic_unit_with_downsampling(x, out_channels)

        for j in range(2, num_units + 1):
            with tf.variable_scope('unit_%d' % j):
                x, y = concat_shuffle_split(x, y)
                x = basic_unit(x)
        x = tf.concat([x, y], axis=3)

    return x


def concat_shuffle_split(x, y):
    with tf.name_scope('concat_shuffle_split'):
        shape = tf.shape(x)
        batch_size = shape[0]
        height, width = shape[1], shape[2]
        depth = x.shape[3].value

        z = tf.stack([x, y], axis=3)  # shape [batch_size, height, width, 2, depth]
        z = tf.transpose(z, [0, 1, 2, 4, 3])
        z = tf.reshape(z, [batch_size, height, width, 2*depth])
        x, y = tf.split(z, num_or_size_splits=2, axis=3)
        return x, y


def basic_unit(x):
    in_channels = x.shape[3].value
    x = slim.conv2d(x, in_channels, (1, 1), stride=1, scope='conv1x1_before')
    x = depthwise_conv(x, kernel=3, stride=1, activation_fn=None, scope='depthwise')
    x = slim.conv2d(x, in_channels, (1, 1), stride=1, scope='conv1x1_after')
    return x


def basic_unit_with_downsampling(x, out_channels=None):
    in_channels = x.shape[3].value
    out_channels = 2 * in_channels if out_channels is None else out_channels

    y = slim.conv2d(x, in_channels, (1, 1), stride=1, scope='conv1x1_before')
    y = depthwise_conv(y, kernel=3, stride=2, activation_fn=None, scope='depthwise')
    y = slim.conv2d(y, out_channels // 2, (1, 1), stride=1, scope='conv1x1_after')

    with tf.variable_scope('second_branch'):
        x = depthwise_conv(x, kernel=3, stride=2, activation_fn=None, scope='depthwise')
        x = slim.conv2d(x, out_channels // 2, (1, 1), stride=1, scope='conv1x1_after')
        return x, y


@tf.contrib.framework.add_arg_scope
def depthwise_conv(
        x, kernel=3, stride=1, padding='SAME',
        activation_fn=None, normalizer_fn=None,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        data_format='NHWC', scope='depthwise_conv'):

    with tf.variable_scope(scope):
        assert data_format == 'NHWC'
        in_channels = x.shape[3].value
        W = tf.get_variable(
            'depthwise_weights',
            [kernel, kernel, in_channels, 1], dtype=tf.float32,
            initializer=weights_initializer
        )
        x = tf.nn.depthwise_conv2d(x, W, [1, stride, stride, 1], padding, data_format='NHWC')
        x = normalizer_fn(x) if normalizer_fn is not None else x  # batch normalization
        x = activation_fn(x) if activation_fn is not None else x  # nonlinearity
        return x



# def inference(image, keep_prob):
#     """
#     Semantic segmentation network definition
#     :param image: input image. Should have values in range 0-255
#     :param keep_prob:
#     :return:
#     """
#     a = tf.layers.conv2d(image, 64, [3, 3], padding='SAME', )
#     print("setting up vgg initialized conv layers ...")

#     mean_pixel = np.mean(mean, axis=(0, 1))


#     processed_image = utils.process_image(image, mean_pixel)


#         annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

#     return tf.expand_dims(annotation_pred, dim=3), conv_t3


def train(loss_val, var_list):
    # optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    optimizer = tf.train.GradientDescentOptimizer(decay_learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def conv(x, ksize, filters_out, stride=1, trainable=True):

    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]

    initializer = tf.contrib.layers.variance_scaling_initializer()
    regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)
    trainable = True

    weights = tf.get_variable('weights',
                              shape=shape,
                              initializer=initializer,
                              regularizer=regularizer,
                              trainable=trainable)

    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')


def decode(feed1, feed2, conv5):

    with tf.variable_scope('feed2'):
        feed2_score = conv(feed2, 1, 2, stride=1)
        # feed2_score = slim.conv2d(feed2, 1, (1, 1), stride=1, scope='feed2_score')

        ###
        feed2_score = tf.layers.batch_normalization(feed2_score, axis=3, center=True, scale=True,
                                                    training=True,
                                                    momentum=BATCH_NORM_MOMENTUM,
                                                    epsilon=BATCH_NORM_EPSILON,
                                                    fused=True)
        ###

        feed2_score = tf.nn.relu(feed2_score)

    with tf.variable_scope('feed3'):

        conv5 = conv(conv5, 1, 2, stride=1)
        # conv5 = slim.conv2d(conv5, 1, (1, 1), stride=1, scope='conv5')
        
        ###
        conv5 = tf.layers.batch_normalization(conv5, axis=3, center=True, scale=True,
                                                training=True,
                                                momentum=BATCH_NORM_MOMENTUM,
                                                epsilon=BATCH_NORM_EPSILON,
                                                fused=True)
        # ###

        conv5 = tf.nn.relu(conv5)

        conv5_shape = conv5.shape.as_list()[1:3]
    # conv5_shape = conv5.get_shape().as_list()[1:3]
    # print (conv5.shape.as_list()[1:3])
    upscore2_upsample = tf.image.resize_images(conv5, (2 * conv5_shape[0], 2 * conv5_shape[1]))

    fuse_feed2 = tf.add(feed2_score, upscore2_upsample)

    with tf.variable_scope('feed1'):

        feed1_score = conv(feed1, 1, 2, stride=1)
        
        ###
        feed1_score = tf.layers.batch_normalization(feed1_score, axis=3, center=True, scale=True,
                                                    training=True,
                                                    momentum=BATCH_NORM_MOMENTUM,
                                                    epsilon=BATCH_NORM_EPSILON,
                                                    fused=True)
        ###

        feed1_score = tf.nn.relu(feed1_score)

    fuse_feed2_shape = fuse_feed2.shape.as_list()[1:3]
    # fuse_feed2_shape = fuse_feed2.get_shape().as_list()[1:3]
    upscore4_upsample = tf.image.resize_images(fuse_feed2, (2 * fuse_feed2_shape[0], 2 * fuse_feed2_shape[1]))

    fuse_feed1 = tf.add(feed1_score, upscore4_upsample)
    

    fuse_feed1_shape = fuse_feed1.shape.as_list()[1:3]
    # fuse_feed1_shape = fuse_feed1.get_shape().as_list()[1:3]
    upscore8_upsample = tf.image.resize_images(fuse_feed1, (8 * fuse_feed1_shape[0], 8 * fuse_feed1_shape[1]))


    annotation_pred = tf.argmax(upscore8_upsample, dimension=3, name="prediction")

    # return segmentation_logits
    return tf.expand_dims(annotation_pred, dim=3), upscore8_upsample



def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, IMAGE_SIZE_H, IMAGE_SIZE_W, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, IMAGE_SIZE_H, IMAGE_SIZE_W, 1], name="annotation")
    # image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    # annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")
    
    precessed_image = (1.0 / 255.0) * tf.to_float(image)  # to [0, 1] range

    
    feed1, feed2, conv5 = shufflenet(precessed_image, is_training = True, num_classes=1000, depth_multiplier='1.0')

    pred_annotation, logits = decode(feed1, feed2, conv5)
    
# ##########3
#     tf.summary.image("input_image", image, max_outputs=2)
#     tf.summary.image("ground_truth", tf.cast(annotation*255, tf.uint8), max_outputs=2)
#     tf.summary.image("pred_annotation", tf.cast(pred_annotation*255, tf.uint8), max_outputs=2)
# ########

    # loss function by junghun "https://stackoverflow.com/questions/44560549/unbalanced-data-and-weighted-cross-entropy"
    # class_weights = tf.constant([1.0,100.0])    
    # weights = tf.reduce_sum(class_weights * onehot_labels, axis=1)
    # unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(onehot_labels, logits)
    # weighted_losses = unweighted_losses * weights
    # loss = tf.reduce_mean(weighted_losses)

    # loss function by junghun "https://stackoverflow.com/questions/40198364/how-can-i-implement-a-weighted-cross-entropy-loss-in-tensorflow-using-sparse-sof/46984951#46984951"
    class_weights = tf.constant([1.0,100.0])
    weights = tf.gather(class_weights, annotation)
    # unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(onehot_labels, logits)
    # weighted_losses = unweighted_losses * weights

# ######
#     loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(annotation, logits, weights))
# #######


    # original loss function (no class weight) 
    # loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
    #                                                                       labels=tf.squeeze(annotation, squeeze_dims=[3]),
    #                                                                       name="entropy")))
    

# #########
#     loss_summary = tf.summary.scalar("entropy", loss)
# #########
#     trainable_var = tf.trainable_variables()
#     if FLAGS.debug:
#         for var in trainable_var:
#             utils.add_to_regularization_and_summary(var)
    
# # #########
# #     train_op = train(loss, trainable_var)
# # ########

    #train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

###########
    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()
###########

    print("Setting up image reader...")
    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
    print(len(train_records))
    print(len(valid_records))

    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size_h': IMAGE_SIZE_H, 'resize_size_w': IMAGE_SIZE_W}
    # image_options = {'resize': True, 'resize_size': IMAGE_SIZE}

###########3
    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.BatchDatset(train_records, image_options)
        validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)
###########
    sess = tf.Session()

    

    # create two summary writers to show training loss and validation loss in the same graph
    # need to create two folders 'train' and 'validation' inside FLAGS.logs_dir
    train_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/train', sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/validation')


    # ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    #if ckpt and ckpt.model_checkpoint_path:

##########
    #
##########
    



    # reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    # reuse_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="ShuffleNetV2")
    # for var in reuse_vars:
    #     if "Conv1" not in var.name :
    #         print(var.name)
    
    #imagenet pretrained model load
    # variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.model_dir + "model.ckpt-1661328",
    #                                                          slim.get_trainable_variables(),
    #                                                          ignore_missing_vars=True)
    # #imagenet pretrained model load

    variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pupil_seg_model_dir + "model.ckpt-100000",
                                                             slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)
    # reuse_vars_dict = dict([(var.name, var) for var in reuse_vars])
    # reuse_vars_dict = dict([(var.name, var) for var in reuse_vars if "batch_norm" not in var.name])

    #print(reuse_vars_dict)


    print("Setting up Saver...")
    # saver = tf.train.Saver(reuse_vars_dict)   
    saver = tf.train.Saver()   

    # saver.restore(sess, ckpt.model_checkpoint_path)
    # saver.restore(sess, "")


    sess.run(tf.global_variables_initializer())
    # print(FLAGS.model_dir + "model.ckpt-1661328")

    variable_restore_op(sess)

    #saver = tf.train.import_meta_graph(FLAGS.model_dir + "model.ckpt-1661328.meta")

    #saver.restore(sess, FLAGS.model_dir + "model.ckpt-1661328")
    print(FLAGS.model_dir + "model.ckpt-1661328")

    # new_saver = tf.train.Saver()

##########3
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation*255, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation*255, tf.uint8), max_outputs=2)
########

######
    loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(annotation, logits, weights))
#######

#########
    loss_summary = tf.summary.scalar("entropy", loss)
#########
    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    
#########
    train_op = train(loss, trainable_var)
    # print("traaaauubausbdusabduabdub",train_op)
########

###########
    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()
###########


    print("Model restored...")

    if FLAGS.mode == "train":
        for itr in xrange(MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            feed_dict = {image: train_images, annotation: train_annotations}
            # feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

            sess.run(train_op, feed_dict=feed_dict)


            if itr % 10 == 0:
                train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                # train_loss = sess.run([loss], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                train_writer.add_summary(summary_str, itr)


            if itr % 500 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                valid_loss, summary_sva = sess.run([loss, loss_summary], feed_dict={image: valid_images, annotation: valid_annotations})
                # valid_loss, summary_sva = sess.run([loss, loss_summary], feed_dict={image: valid_images, annotation: valid_annotations,
                #                                        keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))

                # add validation loss to TensorBoard
                validation_writer.add_summary(summary_sva, itr)
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

    elif FLAGS.mode == "visualize":
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations})
        # pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
        #                                             keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(FLAGS.batch_size):
            utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr))
            utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr))
            utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5+itr))
            print("Saved image: %d" % itr)


if __name__ == "__main__":
    tf.app.run()
