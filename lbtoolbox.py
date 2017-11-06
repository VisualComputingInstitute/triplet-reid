# This file contains select utilities from Lucas Beyer's toolbox, the complete
# toolbox can be found at https://github.com/lucasb-eyer/lbtoolbox.
#
# The content of this file is copyright Lucas Beyer. You may only re-use
# parts of it by keeping the following comment above it:
#
# This is taken from Lucas Beyer's toolbox© found at
#     https://github.com/lucasb-eyer/lbtoolbox
# and may only be redistributed and reused by keeping this notice.

import json
import signal

import numpy as np


def tuplize(what, lists=True, tuplize_none=False):
    """
    If `what` is a tuple, return it as-is, otherwise put it into a tuple.
    If `lists` is true, also consider lists to be tuples (the default).
    If `tuplize_none` is true, a lone `None` results in an empty tuple,
    otherwise it will be returned as `None` (the default).
    """
    if what is None:
        if tuplize_none:
            return tuple()
        else:
            return None

    if isinstance(what, tuple) or (lists and isinstance(what, list)):
        return tuple(what)
    else:
        return (what,)


def create_dat(basename, dtype, shape, fillvalue=None, **meta):
    """ Creates a data file at `basename` and returns a writeable mem-map
        backed numpy array to it.
        Can also be passed any json-serializable keys and values in `meta`.
    """
    # Sadly, we can't just add attributes (flush) to a numpy array,
    # so we need to dummy-subclass it.
    class LBArray(np.ndarray):
        pass

    Xm = np.memmap(basename, mode='w+', dtype=dtype, shape=shape)
    Xa = np.ndarray.__new__(np.ndarray, dtype=dtype, shape=shape, buffer=Xm)
    # Xa = np.ndarray.__new__(LBArray, dtype=dtype, shape=shape, buffer=Xm)
    # Xa.flush = Xm.flush

    if fillvalue is not None:
        Xa.fill(fillvalue)
        Xm.flush()
        # Xa.flush()

    meta.setdefault('dtype', np.dtype(dtype).str)
    meta.setdefault('shape', tuplize(shape))
    json.dump(meta, open(basename + '.json', 'w+'))

    return Xa


def load_dat(basename, mode='r'):
    """ Returns a read-only mem-mapped numpy array to file at `basename`.
    If `mode` is set to `'r+'`, the data can be written, too.
    """
    desc = json.load(open(basename + '.json', 'r'))
    dtype, shape = desc['dtype'], tuplize(desc['shape'])
    Xm = np.memmap(basename, mode=mode, dtype=dtype, shape=shape)
    Xa = np.ndarray.__new__(np.ndarray, dtype=dtype, shape=shape, buffer=Xm)
    #Xa.flush = Xm.flush  # Sadly, we can't just add attributes to a numpy array, need to subclass it.
    return Xa


def create_or_resize_dat(basename, dtype, shape, fillvalue=None, **meta):
    # Not cleanly possible otherwise yet, see https://github.com/numpy/numpy/issues/4198
    try:
        old_desc = json.load(open(basename + '.json', 'r'))
    except:
        return create_dat(basename, dtype, shape, fillvalue, **meta)

    old_dtype, old_shape = old_desc['dtype'], tuplize(old_desc['shape'])

    # Standarize parameters
    new_shape = tuplize(shape)
    new_dtype_str = np.dtype(dtype).str

    # For memory-layout and code-simplicity reasons, we only support growing
    # in the first dimension, which actually covers all my use-cases so far.
    # https://github.com/numpy/numpy/issues/4198#issuecomment-341983443
    assert old_shape[1:] == new_shape[1:], "Can only grow in first dimension! Old: {}, New: {}".format(old_shape, new_shape)
    assert old_dtype == new_dtype_str, "Can't change the dtype! Old: {}, New: {}".format(old_dtype, new_dtype_str)

    # Open the mem-mapped file and reshape it to what's needed.
    Xm = np.memmap(basename, mode='r+', dtype=dtype, shape=old_shape)
    Xm._mmap.resize(Xm.dtype.itemsize * np.product(new_shape))  # BYTES HERE!!

    Xa = np.ndarray.__new__(np.ndarray, dtype=dtype, shape=new_shape, buffer=Xm._mmap, offset=0)
    # Xa.flush = Xm.flush

    if fillvalue is not None:
        Xa[old_shape[0]:] = fillvalue
        Xm._mmap.flush()
        # Xa.flush()

    meta.setdefault('dtype', new_dtype_str)
    meta.setdefault('shape', new_shape)
    json.dump(meta, open(basename + '.json', 'w+'))  # Overwrite the old one.

    return Xa


# Based on an original idea by https://gist.github.com/nonZero/2907502 and heavily modified.
class Uninterrupt(object):
    """
    Use as:
    with Uninterrupt() as u:
        while not u.interrupted:
            # train
    """
    def __init__(self, sigs=(signal.SIGINT,), verbose=False):
        self.sigs = sigs
        self.verbose = verbose
        self.interrupted = False
        self.orig_handlers = None

    def __enter__(self):
        if self.orig_handlers is not None:
            raise ValueError("Can only enter `Uninterrupt` once!")

        self.interrupted = False
        self.orig_handlers = [signal.getsignal(sig) for sig in self.sigs]

        def handler(signum, frame):
            self.release()
            self.interrupted = True
            if self.verbose:
                print("Interruption scheduled...", flush=True)

        for sig in self.sigs:
            signal.signal(sig, handler)

        return self

    def __exit__(self, type_, value, tb):
        self.release()

    def release(self):
        if self.orig_handlers is not None:
            for sig, orig in zip(self.sigs, self.orig_handlers):
                signal.signal(sig, orig)
            self.orig_handlers = None
