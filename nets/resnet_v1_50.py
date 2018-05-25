import tensorflow as tf

from nets.resnet_v1 import resnet_v1_50, resnet_arg_scope

_RGB_MEAN = [123.68, 116.78, 103.94]

def endpoints(image, is_training):
    if image.get_shape().ndims != 4:
        raise ValueError('Input must be of size [batch, height, width, 3]')

    image = image - tf.constant(_RGB_MEAN, dtype=tf.float32, shape=(1,1,1,3))

    with tf.contrib.slim.arg_scope(resnet_arg_scope(batch_norm_decay=0.9, weight_decay=0.0)):
        _, endpoints = resnet_v1_50(image, num_classes=None, is_training=is_training, global_pool=True)

    endpoints['model_output'] = endpoints['global_pool'] = tf.reduce_mean(
        endpoints['resnet_v1_50/block4'], [1, 2], name='pool5')

    return endpoints, 'resnet_v1_50'
