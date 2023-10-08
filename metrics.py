import numpy as np


# Pillai, Arvind, et al. "Personalized Step Counting Using Wearable Sensors: A Domain Adapted LSTM Network Approach."
# arXiv preprint arXiv:2012.08975 (2020).

# Brajdic, Agata, and Robert Harle. "Walk detection and step counting on unconstrained smartphones." Proceedings of
# the 2013 ACM international joint conference on Pervasive and ubiquitous computing. 2013.


def Pillai(y_trues, y_preds, mean=False, sum=False):
    accuracy = 1 - np.abs((y_trues.round() - y_preds.round()) / y_trues.round())
    # accuracy = 1 - np.abs((y_trues - y_preds) / y_trues)
    if sum:
        return np.sum(accuracy)
    elif mean:
        return np.mean(accuracy)
    else:
        return accuracy


def Brajdic(y_trues, y_preds, mean=False, median=False, sum=False):
    # error = (y_trues.round() - y_preds.round()) / y_trues.round() * 100
    error = (y_trues - y_preds) / y_trues * 100
    if sum:
        return np.sum(error)
    elif mean:
        return np.mean(error)
    elif median:
        return np.median(error)
    else:
        return error


def Flores(y_trues, y_preds, mean=False, median=False, sum=False):
    y_trues = y_trues
    y_preds = np.round(y_preds)
    difference = y_trues - y_preds
    undercount = np.abs(np.sum(difference[difference < 0])) / np.sum(y_trues) * 100
    overcount = np.abs(np.sum(difference[difference >= 0])) / np.sum(y_trues) * 100
    error = undercount + overcount
    return error, undercount, overcount


def FloresUnbiased(y_trues, y_preds, mean=False, median=False, sum=False):
    y_trues = y_trues
    y_preds = np.round(y_preds)
    difference = y_trues - y_preds

    undercount = np.abs(np.sum(difference[difference < 0])) / np.sum(y_trues[difference < 0]) * 100
    overcount = np.abs(np.sum(difference[difference >= 0])) / np.sum(y_trues[difference >= 0]) * 100

    error = undercount + overcount

    overall_error = np.abs((y_trues.round() - y_preds.round()) / y_trues.round())
    overall_error = np.mean(overall_error)

    return overall_error, error, undercount, overcount
