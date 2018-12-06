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
from scipy.spatial.distance import cosine

from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Component
from deeppavlov.models.vectorizers.hashing_tfidf_vectorizer import HashingTfIdfVectorizer

logger = get_logger(__name__)


@register("tfidf_dummy_ranker")
class TfidfRanker(Component):
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

    def __call__(self, query_context_id: List[Tuple[str, List[str], List[Any]]]):
        """
        Rank documents and return top n document titles with scores.
        :param questions: queries to search an answer for
        :param n: a number of documents to return
        :return: document ids, document scores
        """

        all_docs = []
        all_scores = []
        all_ids = []

        for query, contexts, indexes in query_context_id:
            q_tfidf = self.vectorizer([query])[0].toarray()
            context_tfidfs = self.vectorizer(contexts)[0].toarray().tolist()

            cos_similairties = [cosine(q_tfidf, context_tfidfs[i]) for i in range(len(context_tfidfs))]

            scores = sorted(cos_similairties, reverse=True)
            indices = np.argsort(cos_similairties)[::-1]
            docs = [contexts[i] for i in indices]
            ids = [indexes[i] for i in indices]

            if self.active:
                docs = docs[:self.top_n]
                scores = scores[:self.top_n]
                ids = ids[:self.top_n]

            all_docs.append(docs)
            all_scores.append(scores)
            all_ids.append(ids)

        return all_docs, all_scores, all_ids
