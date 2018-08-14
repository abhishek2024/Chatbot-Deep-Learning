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

from typing import List, Tuple, Any, Union, Optional
from operator import itemgetter

import numpy as np
from nltk.tokenize import sent_tokenize

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.component import Component
from deeppavlov.skills.odqa.use_sentence_ranker import USESentenceRanker
from deeppavlov.skills.odqa.elmo_ranker import ELMoRanker
from deeppavlov.models.tokenizers.spacy_tokenizer import StreamSpacyTokenizer

logger = get_logger(__name__)


@register("tfhub_paragraph_ranker")
class TFHUBParagraphRanker(Component):

    def __init__(self, ranker: Union[USESentenceRanker, ELMoRanker], tokenizer: Optional[StreamSpacyTokenizer] = None,
                 top_n=10, active: bool = True, sentencize_fn=sent_tokenize, **kwargs):
        """
        :param ranker: an inner ranker, can be USE or ELMo models from tfhub
        :param top_n: top n of document ids to return
        :param active: when is not active, return all doc ids.
        """

        self.top_n = top_n
        self._ranker = ranker
        self._ranker.active = False
        self.sentencize_fn = sentencize_fn
        self.tokenizer = tokenizer
        self.active = active

    def __call__(self, query_context_id: List[Tuple[str, List[str], List[Any]]]):

        """
        todo: optimize query embedding counting (now counts at every iteration)
        """

        all_docs = []
        all_scores = []
        all_ids = []

        for query, contexts, ids in query_context_id:
            query_chunks_pairs = []
            if isinstance(self._ranker, ELMoRanker):
                tokenized_query = self.tokenizer([query])
                tokenized_contexts = self.tokenizer(contexts)
                query_chunks_pairs += [(' '.join(tokenized_query[0]), ' '.join(c)) for c in tokenized_contexts]
            elif isinstance(self._ranker, USESentenceRanker):
                for context in contexts:
                    sentences = self.sentencize_fn(context)
                    query_chunks_pairs.append((query, sentences))
            else:
                raise TypeError(
                    'Unsupported inner ranker type. Select from USESentenceRanker or ElmoRanker')

            _, scores = self._ranker(query_chunks_pairs)
            mean_scores = [np.mean(score) for score in scores]
            text_score_id = list(zip(contexts, mean_scores, ids))
            text_score_id = sorted(text_score_id, key=itemgetter(1), reverse=True)
            if self.active:
                text_score_id = text_score_id[:self.top_n]

            batch_docs = []
            batch_scores = []
            batch_ids = []
            for text, score, doc_id in text_score_id:
                batch_docs.append(text)
                batch_scores.append(score)
                batch_ids.append(doc_id)

            all_docs.append(batch_docs)
            all_scores.append(batch_scores)
            all_ids.append(batch_ids)

        return all_docs, all_scores, all_ids
