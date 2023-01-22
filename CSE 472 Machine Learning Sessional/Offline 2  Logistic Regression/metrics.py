"""
Refer to: https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
"""
import numpy as np


def getStats(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    tp = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    tn = np.sum(np.logical_and(y_true == 0, y_pred == 0))
    fp = np.sum(np.logical_and(y_true == 0, y_pred == 1))
    fn = np.sum(np.logical_and(y_true == 1, y_pred == 0))
    return tp, tn, fp, fn


def accuracy(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return:
    """
    tp, tn, fp, fn = getStats(y_true, y_pred)
    return (tp+tn) / (tp+tn+fp+fn)
    

def precision_score(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    tp, tn, fp, fn = getStats(y_true, y_pred)
    return tp / (tp+fp)


def recall_score(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    tp, tn, fp, fn = getStats(y_true, y_pred)
    return tp / (tp+fn)


def f1_score(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return:
    """
    tp, tn, fp, fn = getStats(y_true, y_pred)
    return (2*tp) / (2*tp+fp+fn)
