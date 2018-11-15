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

from typing import List, Callable, Union, Any, Tuple, Dict
from itertools import chain
from operator import itemgetter

from nltk import sent_tokenize

from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

logger = get_logger(__name__)


@register('document_chunker')
class DocumentChunker(Component):
    """ Make chunks from a document or a list of documents. Don't tear up sentences if needed.
    Args:
        sentencize_fn: a function for sentence segmentation
        keep_sentences: whether to tear up sentences between chunks or not
        tokens_limit: a number of tokens in a single chunk (usually this number corresponds to the squad model limit)
        flatten_result: whether to flatten the resulting list of lists of chunks
    Attributes:
        keep_sentences: whether to tear up sentences between chunks or not
        tokens_limit: a number of tokens in a single chunk
        flatten_result: whether to flatten the resulting list of lists of chunks
    """

    def __init__(self, sentencize_fn: Callable = sent_tokenize, keep_sentences: bool = True,
                 tokens_limit: int = 400, flatten_result: bool = False, *args, **kwargs):
        self._sentencize_fn = sentencize_fn
        self.keep_sentences = keep_sentences
        self.tokens_limit = tokens_limit
        self.flatten_result = flatten_result

    def __call__(self, batch_docs: List[Union[str, List[str]]]) -> List[Union[List[str], List[List[str]]]]:
        """ Make chunks from a batch of documents. There can be several documents in each batch.
        Args:
            batch_docs: a batch of documents / a batch of lists of documents
        Returns:
            chunks of docs, flattened or not
        """

        result = []

        for docs in batch_docs:
            batch_chunks = []
            if isinstance(docs, str):
                docs = [docs]
            for doc in docs:
                doc_chunks = []
                if self.keep_sentences:
                    sentences = sent_tokenize(doc)
                    n_tokens = 0
                    keep = []
                    for s in sentences:
                        n_tokens += len(s.split())
                        if n_tokens > self.tokens_limit:
                            if keep:
                                doc_chunks.append(' '.join(keep))
                                n_tokens = 0
                                keep.clear()
                        keep.append(s)
                    if keep:
                        doc_chunks.append(' '.join(keep))
                    batch_chunks.append(doc_chunks)
                else:
                    split_doc = doc.split()
                    doc_chunks = [split_doc[i:i + self.tokens_limit] for i in
                                  range(0, len(split_doc), self.tokens_limit)]
                    batch_chunks.append(doc_chunks)
            result.append(batch_chunks)

        if self.flatten_result:
            if isinstance(result[0][0], list):
                for i in range(len(result)):
                    flattened = list(chain.from_iterable(result[i]))
                    result[i] = flattened

        return result


@register('string_multiplier')
class StringMultiplier(Component):
    """Make a list of strings from a provided string. A length of the resulting list equals a length
    of a provided reference argument.
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, batch_s: List[str], ref: List[List[str]]) -> List[List[str]]:
        """ Multiply each string in a provided batch of strings.
        Args:
            batch_s: a batch of strings to be multiplied
            ref: a reference to obtain a length of the resulting list
        Returns:
            a multiplied s as list
        """
        res = []

        for s, r in zip(batch_s, ref):
            res.append([s] * len(r))

        return res


@register('text_extractor')
class TextExtractor(Component):
    """Extract text and scores from a ranker output"""

    def __init__(self, **kwargs):
        pass

    def __call__(self, ranker_output: List[List[List[Any]]]):
        TEXT_IDX = 1
        SCORE_IDX = 2
        all_texts = []
        all_scores = []
        for instance in ranker_output:
            texts = list(map(itemgetter(TEXT_IDX), instance))
            scores = list(map(itemgetter(SCORE_IDX), instance))
            all_texts.append(texts)
            all_scores.append(scores)

        return all_texts, all_scores


@register('context_adder')
class ContextAdder(Component):

    def __init__(self, **kwargs):
        pass

    def __call__(self, answers: List[List[Tuple[str, Any]]], contexts: List[List[str]],
                 context_indices: List[List[int]]):

        batch_answer_score_id = []

        for answer, context, index in zip(answers, contexts, context_indices):
            res = []
            sorted_context = [context[j] for j in index]
            for i in range(len(answer)):
                res.append((*answer[i], sorted_context[i]))
            batch_answer_score_id.append(res)

        return batch_answer_score_id


@register('table_context_adder')
class TableContextAdder(Component):

    def __init__(self, **kwargs):
        pass

    def __call__(self, batch_answers: List[List[Tuple[str, Any]]], contexts: List[List[Dict]]):

        batch_answer_score_text = []
        context = contexts[0]
        for instance_answer in batch_answers:
            batch_answer_score_text.append([(ia[0], ia[1], context[ia[2]]) for ia in instance_answer])

        return batch_answer_score_text


# @register('bhge_formatter')
# class BHGEFormatter(Component):
#
#     def __init__(self, **kwargs):
#         pass
#
#     def __call__(self, tables, answers):
#
#         return answers