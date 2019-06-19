# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List

import numpy as np

from deeppavlov.core.common.metrics_registry import register_metric


def mean_reciprocal_rank_at_k(y_true: List[int], y_pred: List[List[np.ndarray]], k: int = 10):
    """
    Calculates mean reciprocal rank at k ranking metric.

    Args:
        y_true: Labels.
        y_predicted: Predictions.
            Each prediction contains ranking score of all ranking candidates for the particular data sample.
            It is supposed that the ranking score for the true candidate goes first in the prediction.

    Returns:
        Mean reciprocal rank at k
    """
    mrr = 0
    num_examples = float(len(y_pred))
    predictions = np.array(y_pred)
    predictions = np.flip(np.lexsort((np.random.random(predictions.shape), predictions), -1), -1)[:, :k]
    for label, prediction in zip(y_true, predictions):
        range_true = range(label)
        for i, el in enumerate(prediction):
            if el in range_true:
                mrr += 1 / (i + 1)
    return float(mrr) / num_examples


@register_metric('mrr@10')
def r_at_10(labels, predictions):
    return mean_reciprocal_rank_at_k(labels, predictions, k=10)
