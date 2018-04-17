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

import json
import re
from functools import lru_cache
from pathlib import Path

from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import pymorphy2
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import download_decompress


@register("kg_ranker")
class KGRanker(Component):

    def __init__(self, url, data_path, data_type='events', n_top=5, *args, **kwargs):
        dir_path = Path(data_path)
        required_files = ['{}.jsonl'.format(data_type)]
        if not dir_path.exists():
            dir_path.mkdir()

        if not all((dir_path / f).exists() for f in required_files):
            download_decompress(url, dir_path)

        self.data = []
        with open(dir_path / '{}.jsonl'.format(data_type), 'r') as fin:
            for line in fin:
                self.data.append(json.loads(line))

        if data_type == 'events':
            self.text_features = ['title', 'description', 'body_text', 'short_title', 'tags', 'tagline']
        elif data_type == 'places':
            self.text_features = ['title', 'address', 'body_text', 'description', 'subway', 'tags']

        self.n_top = n_top
        self.stop_words = set(stopwords.words('russian'))
        self.morph = pymorphy2.MorphAnalyzer()

        self._prepare_tfidf()

    def __call__(self, batch, *args, **kwargs):
        events, scores = [], []
        for utt in batch:
            s = np.dot(self.data_tfidf, self.tfidf.transform([self._preprocess_str(utt)]).T).todense()
            top_events = np.argsort(s, axis=0)[::-1]
            events.append(np.squeeze(top_events.tolist())[:self.n_top].tolist())
            scores.append(np.sort(s, axis=0)[::-1][:self.n_top].tolist())
        return events, scores

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
            f = event[feature_name]
            if isinstance(f, list):
                f = ' <LIST_SEP> '.join(self._preprocess_str(x) for x in f)
            else:
                f = self._preprocess_str(f)
            s += f + ' <SEP> '
        s = s.strip()
        return s