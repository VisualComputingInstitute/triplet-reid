#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import cv2
import pickle
import sys


if len(sys.argv) != 3:
    print("Usage: {} IMAGE_LIST_FILE MODEL_WEIGHT_FILE".format(sys.argv[0]))
    sys.exit(1)

# Specify the path to a Market-1501 image that should be embedded and the location of the weights we provided.
image_list = list(map(str.strip, open(sys.argv[1]).readlines()))
weight_fname = sys.argv[2]



# Setup the pretrained ResNet

#This is based on the Lasagne ResNet-50 example with slight modifications to allow for different input sizes.
#The original can be found at: https://github.com/Lasagne/Recipes/blob/master/examples/resnet50/ImageNet%20Pretrained%20Network%20(ResNet-50).ipynb
import theano
import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import DenseLayer
from lasagne.nonlinearities import rectify, softmax


def build_simple_block(incoming_layer, names,
                       num_filters, filter_size, stride, pad,
                       use_bias=False, nonlin=rectify):
    """Creates stacked Lasagne layers ConvLayer -> BN -> (ReLu)

    Parameters:
    ----------
    incoming_layer : instance of Lasagne layer
        Parent layer

    names : list of string
        Names of the layers in block

    num_filters : int
        Number of filters in convolution layer

    filter_size : int
        Size of filters in convolution layer

    stride : int
        Stride of convolution layer

    pad : int
        Padding of convolution layer

    use_bias : bool
        Whether to use bias in conlovution layer

    nonlin : function
        Nonlinearity type of Nonlinearity layer

    Returns
    -------
    tuple: (net, last_layer_name)
        net : dict
            Dictionary with stacked layers
        last_layer_name : string
            Last layer name
    """
    net = []
    net.append((
            names[0],
            ConvLayer(incoming_layer, num_filters, filter_size, stride, pad,
                      flip_filters=False, nonlinearity=None) if use_bias
            else ConvLayer(incoming_layer, num_filters, filter_size, stride, pad, b=None,
                           flip_filters=False, nonlinearity=None)
        ))

    net.append((
            names[1],
            BatchNormLayer(net[-1][1])
        ))
    if nonlin is not None:
        net.append((
            names[2],
            NonlinearityLayer(net[-1][1], nonlinearity=nonlin)
        ))

    return dict(net), net[-1][0]


def build_residual_block(incoming_layer, ratio_n_filter=1.0, ratio_size=1.0, has_left_branch=False,
                         upscale_factor=4, ix=''):
    """Creates two-branch residual block

    Parameters:
    ----------
    incoming_layer : instance of Lasagne layer
        Parent layer

    ratio_n_filter : float
        Scale factor of filter bank at the input of residual block

    ratio_size : float
        Scale factor of filter size

    has_left_branch : bool
        if True, then left branch contains simple block

    upscale_factor : float
        Scale factor of filter bank at the output of residual block

    ix : int
        Id of residual block

    Returns
    -------
    tuple: (net, last_layer_name)
        net : dict
            Dictionary with stacked layers
        last_layer_name : string
            Last layer name
    """
    simple_block_name_pattern = ['res%s_branch%i%s', 'bn%s_branch%i%s', 'res%s_branch%i%s_relu']

    net = {}

    # right branch
    net_tmp, last_layer_name = build_simple_block(
        incoming_layer, list(map(lambda s: s % (ix, 2, 'a'), simple_block_name_pattern)),
        int(lasagne.layers.get_output_shape(incoming_layer)[1]*ratio_n_filter), 1, int(1.0/ratio_size), 0)
    net.update(net_tmp)

    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], list(map(lambda s: s % (ix, 2, 'b'), simple_block_name_pattern)),
        lasagne.layers.get_output_shape(net[last_layer_name])[1], 3, 1, 1)
    net.update(net_tmp)

    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], list(map(lambda s: s % (ix, 2, 'c'), simple_block_name_pattern)),
        lasagne.layers.get_output_shape(net[last_layer_name])[1]*upscale_factor, 1, 1, 0,
        nonlin=None)
    net.update(net_tmp)

    right_tail = net[last_layer_name]
    left_tail = incoming_layer

    # left branch
    if has_left_branch:
        net_tmp, last_layer_name = build_simple_block(
            incoming_layer, list(map(lambda s: s % (ix, 1, ''), simple_block_name_pattern)),
            int(lasagne.layers.get_output_shape(incoming_layer)[1]*4*ratio_n_filter), 1, int(1.0/ratio_size), 0,
            nonlin=None)
        net.update(net_tmp)
        left_tail = net[last_layer_name]

    net['res%s' % ix] = ElemwiseSumLayer([left_tail, right_tail], coeffs=1)
    net['res%s_relu' % ix] = NonlinearityLayer(net['res%s' % ix], nonlinearity=rectify, name = 'res%s_relu' % ix)

    return net, 'res%s_relu' % ix


def build_model(input_size):
    net = {}
    net['input'] = InputLayer(input_size)
    sub_net, parent_layer_name = build_simple_block(
        net['input'], ['conv1', 'bn_conv1', 'conv1_relu'],
        64, 7, 2, 3, use_bias=True)
    net.update(sub_net)
    net['pool1'] = PoolLayer(net[parent_layer_name], pool_size=3, stride=2, pad=0, mode='max', ignore_border=False)
    block_size = list('abc')
    parent_layer_name = 'pool1'
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1, 1, True, 4, ix='2%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='2%s' % c)
        net.update(sub_net)

    block_size = list('abcd')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='3%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='3%s' % c)
        net.update(sub_net)

    block_size = list('abcdef')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='4%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='4%s' % c)
        net.update(sub_net)

    block_size = list('abc')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='5%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='5%s' % c)
        net.update(sub_net)
    net['pool5'] = PoolLayer(net[parent_layer_name], pool_size=7, stride=1, pad=0,
                             mode='average_exc_pad', ignore_border=False)

    return net


#Setup the original network
resnet = build_model(input_size=(None, 3, 256,128))

#Now we modify the network's final pooling layer and add 2 new layers at the end to predict the 128-dimensional embedding.
#Different input size.
inp = resnet['input']

network_features = resnet['pool5']
network_features.pool_size=(8,4)

#New additional final layer
network = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(
        network_features,
        num_units=1024,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform('relu'),
        b=None))

network_out = lasagne.layers.DenseLayer(
        network,
        num_units=128,
        nonlinearity=None,
        W=lasagne.init.Orthogonal())



#Setup the function to predict the embeddings.
predict_features = theano.function(
            inputs=[inp.input_var],
            outputs=lasagne.layers.get_output(network_out, deterministic=True))


#Set the parameters
with np.load(weight_fname) as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network_out, param_values)



#We subtract the per-channel mean of the "mean image" as loaded from the original ResNet-50 weight dump.
#For simplcity, we just hardcode it here.
im_mean = np.asarray([103.0626238, 115.90288257, 123.15163084], dtype=np.float32)



# a little helper function to create a test-time augmentation batch.
def get_augmentation_batch(image, im_mean):
    #Resize it correctly, as needed by the test time augmentation.
    image = cv2.resize(image, (128+16, 256+32))

    #Change into CHW format
    image = np.rollaxis(image,2)

    #Setup storage for the batch
    batch = np.zeros((10,3,256,128), dtype=np.float32)

    #Four corner crops and the center crop
    batch[0] = image[:,16:-16, 8:-8]    #Center crop
    batch[1] = image[:,   :-32,   :-16] #Top left
    batch[2] = image[:,   :-32, 16:]    #Top right
    batch[3] = image[:, 32:,      :-16] #Bottom left
    batch[4] = image[:, 32:,    16:]    #Bottom right

    #Flipping
    batch[5:] = batch[:5,:,:,::-1]

    #Subtract the mean
    batch = batch-im_mean[None,:,None,None]

    return batch



for image_filename in image_list:
    print(image_filename, end=",")
    sys.stdout.flush()

    image = cv2.imread(image_filename)
    if image is None:
        raise ValueError("Couldn't load image {}".format(image_filename))

    #Setup a batch of images and use the function to predict the embedding.
    batch = get_augmentation_batch(image, im_mean)
    embedding = np.mean(predict_features(batch), axis=0)
    print(','.join(map(str, embedding)))
