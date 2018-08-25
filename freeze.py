#!/usr/bin/env python3
from argparse import ArgumentParser
from importlib import import_module
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_io

import common
from nets import NET_CHOICES
from heads import HEAD_CHOICES


parser = ArgumentParser(description='Train a ReID network.')

parser.add_argument(
    '--checkpoint_name', default='market1501_weights/checkpoint-25000', type=common.readable_directory,
    help='Location of checkpoint to freeze.')

parser.add_argument(
    '--frozen_model_path', default='./encoder_trinet.pb', type=common.writeable_directory,
    help='Location to save or load frozen model.')

parser.add_argument(
    '--model_name', default='resnet_v1_50', choices=NET_CHOICES,
    help='Name of the model to use.')

parser.add_argument(
    '--head_name', default='fc1024_normalize', choices=HEAD_CHOICES,
    help='Name of the head to use.')

parser.add_argument(
    '--embedding_dim', default=128, type=common.positive_int,
    help='Dimensionality of the embedding space.')

parser.add_argument(
    '--net_input_height', default=256, type=common.positive_int,
    help='Height of the input directly fed into the network.')

parser.add_argument(
    '--net_input_width', default=128, type=common.positive_int,
    help='Width of the input directly fed into the network.')

parser.add_argument(
    '--save_graph', action='store_true', default=False,
    help='Whether to save frozen graph for visualization.')

parser.add_argument(
    '--load', action='store_true', default=False,
    help='Whether to load frozen model after saving and benchmark.')

parser.add_argument(
    '--batch_size', default=16, type=common.positive_int,
    help='Batch size of dummy data input.')

parser.add_argument(
    '--runs', default=100, type=common.positive_int,
    help='Number of passes through the network to check speed.')


def save(args):
    """
    Freezes a model checkpoint into a tensorflow pb file.
    Default parameters assume using provided tensorflow checkpoint extracted in root directory.
    Input node name: "input"
    Output node name: "head/out_emb"
    """
    images = tf.placeholder(tf.float32, shape=(
        None, args.net_input_height, args.net_input_width, 3), name='input')

    model = import_module('nets.' + args.model_name)
    head = import_module('heads.' + args.head_name)

    endpoints, body_prefix = model.endpoints(images, is_training=False)
    with tf.name_scope('head'):
        endpoints = head.head(endpoints, args.embedding_dim, is_training=False)

    with tf.Session() as sess:
        tf.train.Saver().restore(sess, args.checkpoint_name)
        output_node_names = ['head/out_emb']

        if args.save_graph:
            summary_writer = tf.summary.FileWriter(logdir='./logs/')
            summary_writer.add_graph(graph=sess.graph)
            print('saved graph')

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            tf.get_default_graph().as_graph_def(),
            output_node_names
        )
        with tf.gfile.GFile(args.frozen_model_path, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print('{} ops in the frozen graph.'.format(len(output_graph_def.node)))


def load(args):
    """
    Check that a frozen model can be loaded correctly.
    Runs speed and memory benchmark.
    """
    # check memory usage of model with session config
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.1
    config.gpu_options.allow_growth = True

    with tf.Session(graph=tf.Graph(), config=config) as sess:
        output_graph_def = tf.GraphDef()
        with open(args.frozen_model_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name='')
        print('{} ops in the frozen graph.'.format(len(output_graph_def.node)))

        in_img = sess.graph.get_tensor_by_name('input:0')
        emb = sess.graph.get_tensor_by_name('head/out_emb:0')

        # benchmark speed with given batch_size
        img_data = np.zeros(
            (args.batch_size, args.net_input_height, args.net_input_width, 3))
        t = time.time()
        total_time = 0
        for i in range(args.runs):
            _ = sess.run(emb, feed_dict={in_img: img_data})
            took = time.time() - t
            total_time += took
            print('runs per second: {:.2f}, time per run: {:.5f}'.format(
                1/took, took))
            t = time.time()
        print('averaged runs per second: {:.2f}, averaged time per run: {:.5f}'.format(
            args.runs/total_time, total_time/args.runs))


def main():
    args = parser.parse_args()
    if not args.load:
        save(args)
    else:
        load(args)


if __name__ == '__main__':
    main()
