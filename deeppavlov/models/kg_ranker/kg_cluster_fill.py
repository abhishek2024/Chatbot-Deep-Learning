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

from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register
from fuzzywuzzy import fuzz
from typing import Union, List, Dict
from collections import defaultdict


@register('kg_cluster_filler')
class KudaGoClusterFiller(Component):
    def __init__(self, slots, threshold, *args, **kwargs):
        self.slots = slots
        self.ngrams = defaultdict(dict)
        for id, vals in slots.items():
            for ans in ['yes', 'no']:
                self.ngrams[id][ans] = defaultdict(list)
                for item in vals[ans]:
                    self.ngrams[id][ans][len(item.split())].append(item)
        self.threshold = threshold

    def __call__(self, last_cluster_id: List[str], slot_history: List[Dict[str, str]],
                 utterance: List[str]) -> List[Dict[str, str]]:
        for i, (utt, clust_id) in enumerate(zip(utterance, last_cluster_id)):
            if clust_id is None:
                continue
            clust_id = clust_id[0]
            splitted_utt = utt.split()
            result = self._infer(splitted_utt, clust_id)
            if result:
                slot_history[i][clust_id] = result
            else:
                slot_history[i][clust_id] = 'unknown'
        return slot_history

    def _infer(self, text: List[str], last_cluster_id: str) -> Union[str, None]:
        n = len(text)
        best_score = 0
        best_candidate = None
        for ans, ngrams in self.ngrams[last_cluster_id].items():
            for window, candidates in ngrams.items():
                for w in range(0, n - window + 1):
                    query = ' '.join(text[w:w + window])
                    if query:
                        for c in candidates:
                            score = fuzz.ratio(c, query)
                            if score > best_score:
                                best_score = score
                                best_candidate = ans

        if best_score >= self.threshold:
            return best_candidate
