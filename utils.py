import numpy as np
import torch


def normalize(signals, lengths):
    maxLength = np.max(lengths)
    newSignals = []

    for i in range(len(signals)):
        x = signals[i, :lengths[i]]
        x = (x - np.min(x)) / np.ptp(x)
        x.resize(maxLength)
        newSignals.append(x)
        del x

    newSignals = np.array(newSignals)

    return newSignals


def flipSignals(signals, lengths):
    flippedSignals = []
    for i in range(signals.shape[0]):
        temp = np.zeros_like(signals[i])
        temp[:lengths[i]] = np.flip(signals[i, :lengths[i]])
        flippedSignals.append(temp)
    return np.array(flippedSignals)
