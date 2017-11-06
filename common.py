""" A bunch of general utilities shared by train/embed/eval """

from argparse import ArgumentTypeError
import os

import numpy as np
import tensorflow as tf

# Commandline argument parsing
###

def check_directory(arg, access=os.W_OK, access_str="writeable"):
    """ Check for directory-type argument validity.

    Checks whether the given `arg` commandline argument is either a readable
    existing directory, or a createable/writeable directory.

    Args:
        arg (string): The commandline argument to check.
        access (constant): What access rights to the directory are requested.
        access_str (string): Used for the error message.

    Returns:
        The string passed din `arg` if the checks succeed.

    Raises:
        ArgumentTypeError if the checks fail.
    """
    path_head = arg
    while path_head:
        if os.path.exists(path_head):
            if os.access(path_head, access):
                # Seems legit, but it still doesn't guarantee a valid path.
                # We'll just go with it for now though.
                return arg
            else:
                raise ArgumentTypeError(
                    'The provided string `{0}` is not a valid {1} path '
                    'since {2} is an existing folder without {1} access.'
                    ''.format(arg, access_str, path_head))
        path_head, _ = os.path.split(path_head)

    # No part of the provided string exists and can be written on.
    raise ArgumentTypeError('The provided string `{}` is not a valid {}'
                            ' path.'.format(arg, access_str))


def writeable_directory(arg):
    """ To be used as a type for `ArgumentParser.add_argument`. """
    return check_directory(arg, os.W_OK, "writeable")


def readable_directory(arg):
    """ To be used as a type for `ArgumentParser.add_argument`. """
    return check_directory(arg, os.R_OK, "readable")


def number_greater_x(arg, type_, x):
    try:
        value = type_(arg)
    except ValueError:
        raise ArgumentTypeError('The argument "{}" is not an {}.'.format(
            arg, type_.__name__))

    if value > x:
        return value
    else:
        raise ArgumentTypeError('Found {} where an {} greater than {} was '
            'required'.format(arg, type_.__name__, x))


def positive_int(arg):
    return number_greater_x(arg, int, 0)


def nonnegative_int(arg):
    return number_greater_x(arg, int, -1)


def positive_float(arg):
    return number_greater_x(arg, float, 0)


def float_or_string(arg):
    """Tries to convert the string to float, otherwise returns the string."""
    try:
        return float(arg)
    except (ValueError, TypeError):
        return arg


# Dataset handling
###


def load_dataset(csv_file, image_root, fail_on_missing=True):
    """ Loads a dataset .csv file, returning PIDs and FIDs.

    PIDs are the "person IDs", i.e. class names/labels.
    FIDs are the "file IDs", which are individual relative filenames.

    Args:
        csv_file (string, file-like object): The csv data file to load.
        image_root (string): The path to which the image files as stored in the
            csv file are relative to. Used for verification purposes.
        fail_on_missing (bool): If one or more files from the dataset are not
            present in the `image_root`, either raise an IOError (if True) or
            remove it from the returned dataset (if False).

    Returns:
        (pids, fids) a tuple of numpy string arrays corresponding to the PIDs,
        i.e. the identities/classes/labels and the FIDs, i.e. the filenames.

    Raises:
        IOError if any one file is missing and `fail_on_missing` is True.
    """
    dataset = np.genfromtxt(csv_file, delimiter=',', dtype='|U')
    pids, fids = dataset.T

    # Check if all files exist
    missing = np.full(len(fids), False, dtype=bool)
    for i, fid in enumerate(fids):
        missing[i] = not os.path.isfile(os.path.join(image_root, fid))

    missing_count = np.sum(missing)
    if missing_count > 0:
        if fail_on_missing:
            raise IOError('Using the `{}` file and `{}` as an image root {}/'
                          '{} images are missing'.format(
                               csv_file, image_root, missing_count, len(fids)))
        else:
            print('[Warning] removing {} missing file(s) from the'
                  ' dataset.'.format(missing_count))
            # We simply remove the missing files.
            fids = fids[np.logical_not(missing)]
            pids = pids[np.logical_not(missing)]

    return pids, fids


def fid_to_image(fid, pid, image_root, image_size):
    """ Loads and resizes an image given by FID. Pass-through the PID. """
    # Since there is no symbolic path.join, we just add a '/' to be sure.
    image_encoded = tf.read_file(tf.reduce_join([image_root, '/', fid]))

    # tf.image.decode_image doesn't set the shape, not even the dimensionality,
    # because it potentially loads animated .gif files. Instead, we use either
    # decode_jpeg or decode_png, each of which can decode both.
    # Sounds ridiculous, but is true:
    # https://github.com/tensorflow/tensorflow/issues/9356#issuecomment-309144064
    image_decoded = tf.image.decode_jpeg(image_encoded, channels=3)
    image_resized = tf.image.resize_images(image_decoded, image_size)

    return image_resized, fid, pid
