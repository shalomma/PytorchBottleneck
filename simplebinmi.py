# Simplified MI computation code from https://github.com/ravidziv/IDNNs
import numpy as np


def get_unique_probs(x):
    uniqueids = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, unique_inverse, unique_counts = np.unique(uniqueids, return_index=False, return_inverse=True, return_counts=True)
    return np.asarray(unique_counts / float(sum(unique_counts))), unique_inverse


def bin_calc_information(labelixs, layerdata, binsize):
    # This is even further simplified, where we use np.floor instead of digitize
    nats2bits = 1.0 / np.log(2)

    def get_h(d):
        digitized = np.floor(d / binsize).astype('int')
        p_ts, _ = get_unique_probs( digitized )
        return -np.sum(p_ts * np.log(p_ts))

    H_LAYER = get_h(layerdata)
    H_LAYER_GIVEN_OUTPUT = 0
    for label, ixs in labelixs.items():
        h = get_h(layerdata[ixs])
        H_LAYER_GIVEN_OUTPUT += ixs.mean() * h
    return nats2bits * H_LAYER, nats2bits * (H_LAYER - H_LAYER_GIVEN_OUTPUT)
