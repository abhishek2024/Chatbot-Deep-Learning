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

from logging import getLogger
from typing import List, Union, Tuple

import numpy as np

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

log = getLogger(__name__)


@register('mltrack_ranking')
class MLtrackRanking(Component):

    def __init__(self,
                 save_path: str,
                 **kwargs) -> None:
        """ Initialize class with given parameters"""

        self.save_path = expand_path(save_path) / "predictions.txt"
        if kwargs["mode"] == "infer":
            self.infer = True
            with open(self.save_path, "w+") as f:
                f.write("")
        else:
            self.infer = False

    def __call__(self, data: Union[np.ndarray, List[List[float]], List[List[int]]],
                 ids: List[str], y_true: List[int],
                 *args, **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Process ranking using probabilities

        Args:
            data: batch of lists of vectors with probability distribution
            y_true: batch of reply relevances
            ids: batch of context ids
            *args:
            **kwargs:

        Returns:
            Tuple of relevance and ranking
        """
        ids = np.reshape(ids, (-1, 6))
        y_true = np.reshape(y_true, (-1, 6))
        data = np.reshape(data, (-1, 6, 3))
        relevance = [np.array([el for el, j in zip(y, i) if j != "0"]) for y, i in zip(y_true, ids)]
        real_data = [np.array([el for el, j in zip(vec, i) if j != "0"]) for vec, i in zip(data, ids)]
        prob = [np.max(el, -1) for el in real_data]
        y_pred = [np.argmax(el, -1) for el in real_data]
        sign = [el-1. for el in y_pred]
        scores = [np.multiply(pr, s) for pr, s in zip(prob, sign)]
        ranking = [np.flip(np.argsort(el), -1) for el in scores]
        if self.infer:
            ans = [i+" "+str(rank) for el, sample in zip(ids, ranking) for i, rank in zip(el, sample)]
            with open(self.save_path, "a") as f:
                f.write("\n".join(ans)+"\n")
        return relevance, ranking

