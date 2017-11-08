import numpy as np


def mean(embs):
    return np.mean(embs, axis=0)


def normalized_mean(embs):
    embs = mean(embs)
    return embs / np.linalg.norm(embs, axis=1, keepdims=True)


AGGREGATORS = {
    'mean': mean,
    'normalized_mean': normalized_mean,
}
