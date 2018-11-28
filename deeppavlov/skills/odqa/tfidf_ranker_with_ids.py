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

from typing import List, Any, Tuple

import numpy as np

from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Component
from deeppavlov.models.vectorizers.hashing_tfidf_vectorizer import HashingTfIdfVectorizer

logger = get_logger(__name__)


@register("tfidf_ranker_with_ids")
class TfidfRankerWithIds(Component):
    """Rank documents according to input strings.
    Args:
        vectorizer: a vectorizer class
        top_n: a number of doc ids to return
        active: whether to return a number specified by :attr:`top_n` (``True``) or all ids
         (``False``)
    Attributes:
        top_n: a number of doc ids to return
        vectorizer: an instance of vectorizer class
        active: whether to return a number specified by :attr:`top_n` or all ids
        index2doc: inverted :attr:`doc_index`
        iterator: a dataset iterator used for generating batches while fitting the vectorizer
    """

    def __init__(self, vectorizer: HashingTfIdfVectorizer, top_n=5, active: bool = True, **kwargs):
        self.top_n = top_n
        self.vectorizer = vectorizer
        self.active = active

    def __call__(self, questions: List[str], batch_ids: List[List[Any]]):
        """
        Rank documents and return top n document titles with scores.
        :param questions: queries to search an answer for
        :param n: a number of documents to return
        :return: document ids, document scores
        """

        batch_docs_scores = []

        q_tfidfs = self.vectorizer(questions)
        for ids, q in zip(batch_ids, q_tfidfs):
            # indices = [self.vectorizer.doc_index[k] for k in ids]
            indices = [3, 6, 8]
            t_tfidfs = np.squeeze(np.array([self.vectorizer.tfidf_matrix[:, i].toarray() for i in indices]))
            batch_docs_scores.append((q * t_tfidfs.transpose())[0])

        return batch_docs_scores
