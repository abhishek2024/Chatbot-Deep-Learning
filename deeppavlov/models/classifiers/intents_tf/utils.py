"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np

from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


def labels2onehot(labels, classes, unknown_label='unknown'):
    """
    Convert labels to one-hot vectors for multi-class multi-label classification
    Args:
        labels: list of samples where each sample is a list of classes which sample belongs with
        classes: list of classes' names

    Returns:
        2d array with one-hot representation of given samples
    """
    onehot = np.zeros((len(labels), len(classes)), dtype=np.int32)

    unknown_idx = None
    if unknown_label in classes:
        unknown_idx = classes.index(unknown_label)
    
    for i, s in enumerate(labels):
        for cl in s:
            if cl in classes:
                onehot[i, classes.index(cl)] = 1
            elif unknown_idx is not None:
                onehot[i, unknown_idx] = 1
            else:
                log.warning('Unknown intent {} detected'.format(intent))

    if unknown_idx is not None:
        onehot[np.sum(onehot, axis=1) < 1, unknown_idx] = 1

    return onehot


def probs2labels(probs, threshold, classes):
    """
    Convert vectors of probabilities to labels using confident threshold

    Args:
        probs: list of samples where each sample is a vector of probabilities
               to belong to a given class
        threshold (float): boundary of probability to belong to a class
        classes: array or list of classes' names

    Returns:
        list of lists of labels for each sample
    """
    classes = np.asarray(classes)
    res = []
    for sample in probs:
        greater_thr = sample > threshold
        if np.sum(greater_thr) > 0:
            res.append(list(classes[greater_thr]))
        else:
            res.append([classes[np.argmax(sample)]])
    return res

