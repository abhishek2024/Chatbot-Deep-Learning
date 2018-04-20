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
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.data.utils import download

from fuzzywuzzy import fuzz
import fastText as fasttext
from typing import Union, List, Dict
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from pathlib import Path
import string
import numpy as np

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


@register('kg_knn_cluster_filler')
class KudaGoKNNClusterFiller(Component):
    def __init__(self, slots, threshold, emb_url, emb_folder, *args, **kwargs):
        self.slots = slots
        self.threshold = threshold
        self.emb_file_name = Path(emb_url).name
        self.emb_folder = expand_path(emb_folder)
        self.emb_url = emb_url
        if not (self.emb_folder / self.emb_file_name).exists():
            download(self.emb_folder / self.emb_file_name, self.emb_url)

        self.emb_folder.mkdir(parents=True, exist_ok=True)
        self.embeddings = fasttext.load_model(str(self.emb_folder / self.emb_file_name))

        self.slots_cls_data = {}

        for slot in self.slots:
            if self.slots[slot]['type'] == 'ClusterSlot':
                self.slots_cls_data[slot] = {}
                for cls in ['yes', 'no']:
                    self.slots_cls_data[slot][cls] = [{'text': el, 'emb': self._sent_to_emb(el)} for el in self.slots[slot][cls]]

    def __call__(self, last_cluster_id, slot_history, utterance):
        for utt, cluster_id, slot_h in zip(utterance, last_cluster_id, slot_history):
            if cluster_id is not None:
                cls, (score, _) = self.knn(utt, cluster_id)
                if score > self.threshold:
                    slot_h[cluster_id] = cls
                else:
                    slot_h[cluster_id] = 'unknown'
        return slot_history

    def _preprocess_text(self, s):
        s = ''.join(filter(lambda x: x not in string.punctuation, s.lower()))
        s_tok = self.tokenizer.tokenize(s)
        return s_tok

    def _sent_to_emb(self, sent):
        """
        computes mean word embedding of sentence filtering punctuation and stop words
        Args:
            sent: sentence
        Returns:
            mean word embedding
        """
        sent = ''.join(filter(lambda x: x not in string.punctuation, sent.lower()))
        words = word_tokenize(sent)
        embds = [self.embeddings.get_word_vector(w) / np.linalg.norm(self.embeddings.get_word_vector(w)) for w in words]
        if len(embds) == 0:
            return np.zeros_like(self.embeddings['тест'])
        return np.mean(np.stack(embds), axis=0)

    @staticmethod
    def _cosine_distance(a, b):
        if np.linalg.norm(a) * np.linalg.norm(b) == 0:
            return 0.0
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def knn(self, sent, slot, k=3):
        assert k % 2 != 0
        sent_embedded = self._sent_to_emb(sent)
        scores = []
        for cl in self.slots_cls_data[slot]:
            for cl_s in self.slots_cls_data[slot][cl]:
                score = self._cosine_distance(cl_s['emb'], sent_embedded)
                scores.append((cl, cl_s['text'], score))
        scores = list(sorted(scores, key=lambda x: x[2], reverse=True))
        votes_count = {cl: len(list(filter(lambda x: x[0] == cl, scores[:k]))) for cl in self.slots_cls_data[slot]}
        max_class, max_votes = None, 0
        for cl in votes_count:
            if max_votes < votes_count[cl]:
                max_votes = votes_count[cl]
                max_class = cl
        return max_class, max(map(lambda x: (x[2], x[1]), filter(lambda x: x[0] == max_class, scores)))