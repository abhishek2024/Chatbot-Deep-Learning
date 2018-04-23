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

import re
from nltk import word_tokenize
import pymorphy2
from nltk.corpus import stopwords
from functools import lru_cache
from pathlib import Path
from gensim.models import KeyedVectors

from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import download
from deeppavlov.core.commands.utils import expand_path


@register('kg_wmd')
class WMD(Component):
    def __init__(self,
                 data,
                 emb_folder,
                 emb_url,
                 threshold=75,
                 filter_stop=False,
                 n_last_utts=None,
                 **kwargs):
        """ Word mover distance

        Args:
            data: KudaGo data dump
            data_type: 'events' or 'places' or 'places_events'
            threshold: relative threshold from 0 to 100, reasonable values 70-90
        """
        self._n_last_utts = n_last_utts
        self.emb_file_name = Path(emb_url).name
        self.emb_folder = expand_path(emb_folder)
        self.emb_url = emb_url
        if not (self.emb_folder / self.emb_file_name).exists():
            download(self.emb_folder / self.emb_file_name, self.emb_url)

        self.places_events = data['places_events']
        self.embedder = KeyedVectors.load_word2vec_format(str(self.emb_folder / self.emb_file_name))
        self.lemmatizer = pymorphy2.MorphAnalyzer()
        self.stopwords = stopwords.words('russian')
        self.tokenizer = word_tokenize
        self._filter_stop = filter_stop
        self.id_key = 'local_id'
        self.threshold = threshold
        self._negation_words = ['без', 'не']

    @lru_cache(maxsize=1024)
    def _get_lemma(self, word):
        return self.lemmatizer.parse(word)[0].normal_form

    def preprocess(self, line):
        line = line.lower()
        line = re.sub(r'<.*?>', '', line)
        tokens = self.tokenizer(line)
        if self._filter_stop:
            tokens = [token for token in tokens if token not in self.stopwords]
        tokens_lemmas = [self._get_lemma(token) for token in tokens]
        return tokens_lemmas

    def mathch_wmd(self, u_toks, t_toks):
        return self.embedder.wmdistance(u_toks, t_toks)

    def match(self, u_toks, t_toks):
        if len(t_toks) < len(u_toks):
            seq_short = t_toks
            seq_long = u_toks
            u_long = True
        else:
            seq_short = u_toks
            seq_long = t_toks
            u_long = False
        ls = len(seq_short)
        ll = len(seq_long)

        ratios = []
        for shift in range(ll - ls + 1):
            if shift > 0:
                if u_long:
                    negation = -1 if seq_long[shift - 1] in self._negation_words else 1
                else:
                    negation = -1 if any(tok in self._negation_words for tok in seq_short) else 1
            else:
                negation = 1
            ratios.append((self.mathch_wmd(seq_short, seq_long[shift: shift + ls]), negation))
        distance, negation = min(ratios, key=lambda x: x[0])
        return distance, negation

    def __call__(self, utt_batch):
        for utt in utt_batch:
            if self._n_last_utts is not None:
                utt = self._prepare_utt_history(utt)
            utt_tokens = self.preprocess(utt)
            events_scores = []
            for event in self.places_events:
                event_description_tokens = self.preprocess(event['description'])
                distance= self.mathch_wmd(utt_tokens, event_description_tokens)
                events_scores.append((event[self.id_key], 1 / distance, ))
        return events_scores

    def _prepare_utt_history(self, utt_history):
        return ' <UTT_SEP> '.join([utt for utt in utt_history[-self._n_last_utts:]])
