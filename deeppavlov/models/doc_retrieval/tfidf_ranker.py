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

from logging import getLogger
from typing import List, Any, Tuple

import numpy as np
import scipy as sp

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Component
from deeppavlov.models.vectorizers.hashing_tfidf_vectorizer import HashingTfIdfVectorizer

logger = getLogger(__name__)


@register("tfidf_ranker")
class TfidfRanker(Component):
    """Rank documents according to input strings.

    Args:
        vectorizer: a vectorizer class
        top_n: a number of doc ids to return
        active: whether to return a number specified by :attr:`top_n` (``True``) or all ids
         (``False``)
        cosine: to use cosine tfidf similarity

    Attributes:
        top_n: a number of doc ids to return
        vectorizer: an instance of vectorizer class
        active: whether to return a number specified by :attr:`top_n` or all ids
        index2doc: inverted :attr:`doc_index`
        iterator: a dataset iterator used for generating batches while fitting the vectorizer
        norms: l-2 norms of tfidf documents vectors

    """

    def __init__(self, vectorizer: HashingTfIdfVectorizer, top_n: int = 5,  active: bool = True, cosine: bool = False,
                 **kwargs):

        self.top_n = top_n
        self.vectorizer = vectorizer
        self.active = active
        self.cosine = cosine
        if self.cosine:
            norms = sp.sparse.linalg.norm(vectorizer.tfidf_matrix, axis=0)
            # fix zero norm docs
            self.norms = (norms == 0) * 1 + norms

    def __call__(self, questions: List[str]) -> Tuple[List[Any], List[float]]:
        """Rank documents and return top n document titles with scores.

        Args:
            questions: list of queries used in ranking

        Returns:
            a tuple of selected doc ids and their scores
        """

        batch_doc_ids, batch_docs_scores = [], []

        q_tfidfs = self.vectorizer(questions)

        for q_tfidf in q_tfidfs:
            scores = q_tfidf * self.vectorizer.tfidf_matrix
            if self.cosine:
                q_norm = np.sqrt(np.sum(q_tfidf.multiply(q_tfidf), axis=1)).sum()
                q_norm = q_norm if q_norm > 0 else 1.0
                denominator = q_norm * self.norms
                scores = np.squeeze(np.array(scores / denominator))
            else:
                scores = np.squeeze(scores.toarray() + 0.0001)  # add a small value to eliminate zero scores

            if self.active:
                top_n = self.top_n
            else:
                top_n = len(self.vectorizer.doc_index)

            if top_n >= len(scores):
                o = np.argpartition(-scores, len(scores) - 1)[0:top_n]
            else:
                o = np.argpartition(-scores, top_n)[0:top_n]
            o_sort = o[np.argsort(-scores[o])]

            doc_scores = scores[o_sort]
            doc_ids = [self.vectorizer.index2doc[i] for i in o_sort]
            batch_doc_ids.append(doc_ids)
            batch_docs_scores.append(doc_scores)

        return batch_doc_ids, batch_docs_scores
