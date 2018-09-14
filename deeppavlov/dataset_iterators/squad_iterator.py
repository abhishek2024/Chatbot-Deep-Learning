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


from typing import Dict, Any, List, Tuple

import random

import nltk
import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator


@register('squad_iterator')
class SquadIterator(DataLearningIterator):
    """ SquadIterator allows to iterate over examples in SQuAD-like datasets.
    SquadIterator is used to train :class:`~deeppavlov.models.squad.squad.SquadModel`.

    It extracts ``context``, ``question``, ``answer_text`` and ``answer_start`` position from dataset.
    Example from a dataset is a tuple of ``(context, question)`` and ``(answer_text, answer_start)``

    Attributes:
        train: train examples
        valid: validation examples
        test: test examples

    """

    def split(self, *args, **kwargs) -> None:
        for dt in ['train', 'valid', 'test']:
            setattr(self, dt, self._extract_cqas(getattr(self, dt)))

    @staticmethod
    def _extract_cqas(data: Dict[str, Any]) -> List[Tuple[Tuple[str, str], Tuple[List[str], List[int]]]]:
        """ Extracts context, question, answer, answer_start from SQuAD data

        Args:
            data: data in squad format

        Returns:
            list of (context, question), (answer_text, answer_start)
            answer text and answer_start are lists

        """
        cqas = []
        if data:
            for article in data['data']:
                for par in article['paragraphs']:
                    context = par['context']
                    for qa in par['qas']:
                        q = qa['question']
                        ans_text = []
                        ans_start = []
                        if len(qa['answers']) == 0:
                            # squad 2.0 has questions without an answer
                            cqas.append(((context, q), ([''], [-1])))
                        else:
                            for answer in qa['answers']:
                                ans_text.append(answer['text'])
                                ans_start.append(answer['answer_start'])
                            cqas.append(((context, q), (ans_text, ans_start)))
        return cqas


@register('squad_noans_iterator')
class SquadNoAnsIterator(SquadIterator):
    def split(self, *args, **kwargs):
        squad_qas = {}
        rate = kwargs.get('noans_rate', 0.3)
        for dt in ['train', 'valid', 'test']:
            squad_qas[dt] = self._extract_cqas(getattr(self, dt))

        squad_qas_noans = {}
        for dt in ['train', 'valid', 'test']:
            squad_qas_noans[dt] = self._extract_cqas_noans(getattr(self, dt), rate)

        for dt in ['train', 'valid', 'test']:
            setattr(self, dt, squad_qas[dt] + squad_qas_noans[dt])

    @staticmethod
    def _extract_cqas_noans(data, rate=1.0, qc_rate=0.3):
        """
        Adds random questions with no answer to SQuAD.
        """
        cqas = []
        questions = []
        if data:
            for article in data['data']:
                for par in article['paragraphs']:
                    for qa in par['qas']:
                        questions.append(qa['question'])

            for article in data['data']:
                for par in article['paragraphs']:
                    context = par['context']
                    for qa in par['qas']:
                        if random.random() < rate:
                            if random.random() < qc_rate:
                                # add random question
                                q = random.sample(questions, k=1)[0]
                                ans_text = ['']
                                ans_start = [-1]
                                cqas.append(((context, q), (ans_text, ans_start)))
                            else:
                                # add context without answers
                                q = qa['question']
                                ans_text = ['']
                                ans_start = [-1]
                                new_context = ''
                                for sent in nltk.sent_tokenize(context):
                                    if not any(ans['text'].lower() in sent.lower() for ans in qa['answers']):
                                        new_context += sent + ' '
                                new_context = new_context.strip()
                                if new_context != '':
                                    cqas.append(((new_context, q), (ans_text, ans_start)))
        return cqas


@register('squad_scorer_iterator')
class SquadScorerIterator(DataLearningIterator):
    def split(self, *args, **kwargs):
        for dt in ['train', 'valid', 'test']:
            setattr(self, dt, getattr(self, dt))


@register('multi_squad_iterator')
class MultiSquadIterator(DataLearningIterator):

    def __init__(self, data, seed: int = None, shuffle: bool = True, *args, **kwargs):
        self.rate = kwargs.get('with_answer_rate', 0.666)
        super().__init__(data, seed, shuffle, *args, **kwargs)

    def gen_batches(self, batch_size: int, data_type: str = 'train',
                    shuffle: bool = None):
        # TODO: implement

        if self.shuffle:
            random.shuffle(self.data[data_type])

        data = self.data[data_type]
        data_len = len(data)

        for i in range((data_len - 1) // batch_size + 1):
            batch = []
            for j in range(i * batch_size, (i+1) * batch_size):
                if data_len <= j:
                    break
                q = data[j]['question']
                contexts = data[j]['contexts']
                ans_contexts = [c for c in contexts if len(c['answer']) > 0]
                noans_contexts = [c for c in contexts if len(c['answer']) == 0]
                context = None
                # sample context with answer or without answer
                if np.random.rand() < self.rate or len(noans_contexts) == 0:
                    # select random context with answer
                    context = random.choice(ans_contexts)
                else:
                    # select random context without answer
                    # prob ~ context tfidf score
                    noans_scores = np.array(list(map(lambda x: x['score'], noans_contexts)))
                    noans_scores = noans_scores / np.sum(noans_scores)
                    context = noans_contexts[np.argmax(np.random.multinomial(1, noans_scores))]

                answer_text = [ans['text'] for ans in context['answer']] if len(context['answer']) > 0 else ['']
                answer_start = [ans['answer_start'] for ans in context['answer']] if len(context['answer']) > 0 else [-1]
                batch.append(((context['context'], q), (answer_text, answer_start)))
            yield tuple(zip(*batch))

    def get_instances(self, data_type: str = 'train'):
        data_examples = []
        for qcas in self.data[data_type]:  # question, contexts, answers
            question = qcas['question']
            for context in qcas['contexts']:
                answer_text = list(map(lambda x: x['text'], context['answer']))
                answer_start = list(map(lambda x: x['answer_start'], context['answer']))
                data_examples.append(((context['context'], question), (answer_text, answer_start)))
        return tuple(zip(*data_examples))


@register('ner_squad_iterator')
class NerSquadIterator(DataLearningIterator):

    def split(self, *args, **kwargs) -> None:
        for dt in ['train', 'valid', 'test']:
            setattr(self, dt, self._extract_cqas(getattr(self, dt)))

    @staticmethod
    def _extract_cqas(data: Dict[str, Any]) -> List[Tuple[Tuple[str, str], Tuple[List[str], List[int]]]]:
        cqas = []
        if data:
            for article in data['data']:
                for par in article['paragraphs']:
                    context = par['context']
                    c_processed = par['context_clean']
                    c_tokens = par['context_tokens']
                    c_chars = par['context_chars']
                    spans = par['spans']
                    r2p = par['r2p']
                    p2r = par['p2r']
                    ner_tags = par['ner_tags']
                    for qa in par['qas']:
                        q = qa['question']
                        q_processed = qa['question_clean']
                        q_tokens = qa['question_tokens']
                        q_chars = qa['question_chars']
                        q_ner_tags = qa['ner_tags']

                        x = (context, c_processed, c_tokens, c_chars, ner_tags,
                             q, q_processed, q_tokens, q_chars, q_ner_tags,
                             spans, r2p, p2r)

                        ans_text = []
                        ans_start = []
                        if len(qa['answers']) == 0:
                            # squad 2.0 has questions without an answer
                            cqas.append((x, ([''], [-1])))
                        else:
                            for answer in qa['answers']:
                                ans_text.append(answer['text'])
                                ans_start.append(answer['answer_start'])
                            cqas.append((x, (ans_text, ans_start)))
        return cqas
