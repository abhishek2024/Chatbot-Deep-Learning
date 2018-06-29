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

from typing import List, Tuple, Any
from operator import itemgetter

import numpy as np
from scipy.sparse import csr_matrix
import json
from pathlib import Path
from nltk import sent_tokenize

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.component import Component
from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.core.commands.utils import expand_path

logger = get_logger(__name__)


@register("squad_paragraph_ranker")
class SquadParagraphRanker(Component):

    def __init__(self, ranker_config=None, top_n=10, type='paragraph', **kwargs):
        """

        Args:
            ranker: squad ranker model
            top_n: number of documents to return
            mode: paragraph or sentences
        """
        self.ranker_config = expand_path(Path(ranker_config))
        self.ranker = build_model_from_config(json.load(self.ranker_config.open()))
        self.top_n = top_n
        self.type = type

    def __call__(self, query_context_id: List[Tuple[str, List[str], List[Any]]]):

        all_docs = []
        all_scores = []
        all_ids = []

        for query, contexts, ids in query_context_id:

            scores = []
            for context in contexts:
                if self.type == 'paragraph':
                    _, _, score, _ = self.ranker([(context, query)])[0]
                elif self.type == 'sentences':
                    sentences = sent_tokenize(context)
                    score = []
                    for sent in sentences:
                        _, _, s, _ = self.ranker([(sent, query)])[0]
                        score.append(s)
                    score = np.max(score)
                else:
                    raise RuntimeError('Unsupported type: {}'.format(self.mode))

                scores.append(score)

            texts = contexts
            text_score_id = list(zip(texts, scores, ids))
            text_score_id = sorted(text_score_id, key=itemgetter(1), reverse=True)

            text_score_id = text_score_id[:self.top_n]
            batch_docs = [text for text, score, doc_id in text_score_id]
            batch_scores = [score for text, score, doc_id in text_score_id]
            batch_ids = [doc_id for text, score, doc_id in text_score_id]
            all_docs.append(batch_docs)
            all_scores.append(batch_scores)
            all_ids.append(batch_ids)

        return all_docs, all_scores, all_ids