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


@register("tfidf_ranker")
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

    def __call__(self, questions: List[str]):
        """
        Rank documents and return top n document titles with scores.
        :param questions: queries to search an answer for
        :param n: a number of documents to return
        :return: document ids, document scores
        """

        batch_doc_ids, batch_docs_scores = [], []

        q_tfidfs = self.vectorizer(questions)

        for q_tfidf in q_tfidfs:
            scores = q_tfidf * self.vectorizer.tfidf_matrix
            scores = np.squeeze(
                scores.toarray() + 0.0001)  # add a small value to eliminate zero scores

            # For cosine distance:
            # scores = cosine_similarity(q_tfidf, self.tfidf_matrix.T)
            # scores = scores + 0.0001
            # scores = scores[0]

            # neg_idx = random.randint(1000, 969792)

            if self.active:
                thresh = self.top_n
            else:
                thresh = len(self.vectorizer.doc_index)

            if thresh >= len(scores):
                o = np.argpartition(-scores, len(scores) - 1)[0:thresh]
                # neg = np.argpartition(-scores, len(scores) - 1)[neg_idx]
            else:
                o = np.argpartition(-scores, thresh)[0:thresh]
                # neg = np.argpartition(-scores, thresh)[neg_idx]
            o_sort = o[np.argsort(-scores[o])]

            doc_scores = scores[o_sort]
            doc_ids = [self.vectorizer.index2doc[i] for i in o_sort]

            # neg = self.index2doc[neg]
            # doc_ids.append(neg)
            # doc_scores = np.append(doc_scores, 0.0001)

            batch_doc_ids.append(doc_ids)
            batch_docs_scores.append(doc_scores)

        return batch_doc_ids, batch_docs_scores
