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
from deeppavlov.core.common.log import get_logger
from fuzzywuzzy import fuzz
from typing import Union, List, Dict
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer

log = get_logger(__name__)

@register('kg_cluster_filler')
class KudaGoClusterFiller(Component):
    def __init__(self, slots, threshold, *args, **kwargs):
        self.slots = slots
        self.tokenizer = RegexpTokenizer(r'\w+')
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
            splitted_utt = self._preprocess_text(utt)
            result = self._infer(splitted_utt, clust_id)
            if result:
                slot_history[i][clust_id] = result
            else:
                slot_history[i][clust_id] = 'unknown'
        return slot_history

    def _preprocess_text(self, s):
        s = s.lower()
        s_tok = self.tokenizer.tokenize(s)
        return s_tok

    def _infer(self, tokens: List[str], last_cluster_id: str) -> Union[str, None]:
        num_tokens = len(tokens)
        best_score = 0
        best_candidate = None
        for ans, ngrams in self.ngrams[last_cluster_id].items():
            for num_cand_tokens, candidates in ngrams.items():
                for w in range(0, num_tokens - num_cand_tokens + 1):
                    query = ' '.join(tokens[w: w + num_cand_tokens])
                    if query:
                        for c in candidates:
                            score = fuzz.ratio(c, query)
                            if score > best_score:
                                best_score = score
                                best_candidate = ans

        log.debug("best candidate = {}, best score = {}".format(best_candidate, best_score))
        if best_score >= self.threshold:
            return best_candidate
