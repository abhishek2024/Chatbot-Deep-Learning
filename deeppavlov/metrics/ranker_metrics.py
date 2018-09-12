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
from itertools import zip_longest

import numpy as np

from deeppavlov.core.common.metrics_registry import register_metric


@register_metric('ranker_recall')
def recall(y_true, y_predicted):

    def grouper(iterable, n, fillvalue=0):
        assert len(iterable) % n == 0
        args = [iter(iterable)] * n
        return zip_longest(*args, fillvalue=fillvalue)

    preds = [list(x) for x in grouper(y_predicted, 6)]
    len_y_true = len(y_true) // 6
    y_pred = [1 if np.argmax(el) == 0 else 0 for el in preds]
    return sum(y_pred) * 100 / len_y_true

