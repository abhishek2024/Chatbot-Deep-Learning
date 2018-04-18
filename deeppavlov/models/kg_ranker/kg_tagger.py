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

from collections import defaultdict
from fuzzywuzzy import fuzz
from nltk import word_tokenize
import pymorphy2
from nltk.corpus import stopwords
from functools import lru_cache

from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register


@register('kg_tagger')
class LeveTagger(Component):
    def __init__(self, data, data_type='places_events', threshold=75, **kwargs):
        """ Fuzzy Levenshtein tagger for finding

        Args:
            data: KudaGo data dump
            data_type: 'events' or 'places' or 'places_events'
            threshold: relative threshold from 0 to 100, reasonable values 70-90
        """

        variations = data[f'{data_type}_variations'].items()

        self.lemmatizer = pymorphy2.MorphAnalyzer()
        self.stopwords = stopwords.words('russian')
        self.tokenizer = word_tokenize
        self.tags = []
        for tag, v_list in variations:
            for v in v_list:
                self.tags.append([tag, v])
        self.tags_by_len = self._get_tags_by_len(self.tags)
        self.threshold = threshold

    @lru_cache(maxsize=1024)
    def _get_lemma(self, word):
        return self.lemmatizer.parse(word)[0].normal_form

    def preprocess(self, line):
        line = line.lower()
        tokens = self.tokenizer(line)
        tokens = [token for token in tokens if token not in self.stopwords]
        tokens_lemmas = [self._get_lemma(token) for token in tokens]
        return tokens_lemmas

    def _get_tags_by_len(self, tags):
        tags_by_len = defaultdict(list)
        for tag, tag_val in tags:
            tokens = self.preprocess(tag_val)
            tags_by_len[len(tokens)].append([tag, tokens])
        return tags_by_len

    def match(self, u_toks, t_toks):
        if len(t_toks) < len(u_toks):
            seq_short = t_toks
            seq_long = u_toks
        else:
            seq_short = u_toks
            seq_long = t_toks
        ls = len(seq_short)
        ll = len(seq_long)

        phrase_short = ' '.join(seq_short)
        ratios = []
        for shift in range(ll - ls + 1):
            phrase_long = ' '.join(seq_long[shift: shift + ls])
            ratios.append(fuzz.token_sort_ratio(phrase_short, phrase_long))
        return ratios

    def __call__(self, utt_batch):
        responses = []
        for utt in utt_batch:
            utt_tokens = self.preprocess(utt)
            retrieved_tags = []
            for l in self.tags_by_len:
                for tag, tag_tokens in self.tags_by_len[l]:

                    scores = self.match(utt_tokens, tag_tokens)
                    score = max(scores)
                    if score > self.threshold:
                        retrieved_tags.append([tag, score])
            tags_scores = {}
            for tag, score in retrieved_tags:
                tags_scores[tag] = max(score, tags_scores.get(tag, 0))
            responses.append(tags_scores)
        return responses
