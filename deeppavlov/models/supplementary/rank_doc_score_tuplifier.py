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
from typing import List, Any
from operator import itemgetter

import numpy as np

from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register


@register("rank_doc_score_tuplifier")
class RankDocScoreIdTuplifier(Component):

    def __init__(self, sort_result=False, *args, **kwargs):
        self.sort_result = sort_result

    def __call__(self, docs: List[List[str]], scores: List[np.array], ids: List[Any], title_scores: List[np.array],
                 *args, **kwargs):
        all_results = []
        for triple in zip(docs, scores, ids, title_scores):
            tuples = list(zip(triple[0], triple[1], triple[2], triple[3]))
            if self.sort_result:
                tuples.sort(key=itemgetter(1), reverse=True)
            result = []
            for i, item in enumerate(tuples, 1):
                item = list(item)
                item.insert(0, i)
                item = tuple(item)
                result.append(item)
            all_results.append(result)
        return all_results
