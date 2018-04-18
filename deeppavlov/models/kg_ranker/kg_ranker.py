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
from functools import lru_cache

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import pymorphy2
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register

nltk.download('stopwords')
nltk.download('punkt')


@register("kg_ranker")
class KGRanker(Component):

    def __init__(self, data, data_type='places_events', n_top=5, *args, **kwargs):
        self.data = data[data_type]

        self.tags_variations = {x: data['{}_variations'.format(x)] for x in data_type.split('_')}

        self.text_features = ['title', 'description', 'body_text', 'short_title', 'tags', 'tagline', 'address', 'subway']

        self.n_top = n_top
        self.stop_words = set(stopwords.words('russian'))
        self.morph = pymorphy2.MorphAnalyzer()
        self.id_key = 'local_id' if data_type == 'places_events' else 'id'
        self._prepare_tfidf()

    def __call__(self, batch, *args, **kwargs):
        batch_events, batch_scores = [], []
        for utt in batch:
            s = np.dot(self.data_tfidf, self.tfidf.transform([self._preprocess_str(utt)]).T).todense()
            top_events = np.argsort(s, axis=0)[::-1]
            events = np.squeeze(top_events.tolist()).tolist()
            scores = np.squeeze(np.sort(s.tolist(), axis=0)[::-1]).tolist()
            if self.n_top != -1:
                events = events[:self.n_top]
                scores = scores[:self.n_top]
            batch_events.append(list(map(lambda x: self.data[x][self.id_key], events)))
            batch_scores.append(scores)
        return batch_events, batch_scores

    def _prepare_tfidf(self):
        self.data_texts = [self._get_text(event, self.text_features) for event in tqdm(self.data)]
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2)).fit(self.data_texts)
        self.data_tfidf = self.tfidf.transform(self.data_texts)

    @lru_cache(maxsize=512)
    def _get_lemma(self, word):
        return self.morph.parse(word)[0].normal_form

    def _preprocess_str(self, s):
        s = re.sub(r'<.*?>', '', s)
        s = s.lower()
        s_tok = word_tokenize(s)
        s_lemm = list(map(lambda x: self._get_lemma(x), s_tok))
        s_filtered = list(filter(lambda x: x not in self.stop_words, s_lemm))
        return ' '.join(s_filtered)

    def _get_text(self, event, text_features):
        s = ''
        for feature_name in text_features:
            if feature_name in event:
                f = event[feature_name]
                if feature_name == 'tags':
                    tag_to_str = []
                    for tag in f:
                        tag_variations = self.tags_variations.get(tag, [])
                        tag_str = ' <TAG_VAR_SEP> '.join([self._preprocess_str(x) for x in tag_variations])
                        if len(tag_variations) == 0:
                            tag_str = self._preprocess_str(tag)
                        tag_to_str.append(tag_str)
                    f = ' <TAG_SEP> '.join(tag_to_str)
                elif isinstance(f, list):
                    f = ' <LIST_SEP> '.join(self._preprocess_str(x) for x in f)
                else:
                    f = self._preprocess_str(f)
                s += f + ' <SEP> '
        s = s.strip()
        return s
