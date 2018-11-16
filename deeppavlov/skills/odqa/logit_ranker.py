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

from typing import List, Union, Tuple, Any
from operator import itemgetter
from itertools import chain

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.estimator import Component
from deeppavlov.core.common.chainer import Chainer

logger = get_logger(__name__)


@register("logit_ranker")
class LogitRanker(Component):
    """Select best answer using squad model logits. Make several batches for a single batch, send each batch
     to the squad model separately and get a single best answer for each batch.
     Args:
        squad_model: a loaded squad model
        batch_size: batch size to use with squad model
     Attributes:
        squad_model: a loaded squad model
        batch_size: batch size to use with squad model
    """

    def __init__(self, top_n: 3, squad_model: Union[Chainer, Component], batch_size: int = 10,
                 filter_noans: bool = False, **kwargs):
        self.squad_model = squad_model
        self.batch_size = batch_size
        self.top_n = top_n
        self.filter_noans = filter_noans

    def __call__(self, contexts_batch: List[List[str]], questions_batch: List[List[str]]) -> \
            Tuple[List[List[Tuple[str, Any]]], List[List[int]]]:
        """
        Sort obtained results from squad reader by logits and get the answer with a maximum logit.
        Args:
            contexts_batch: a batch of contexts which should be treated as a single batch in the outer JSON config
            questions_batch: a batch of questions which should be treated as a single batch in the outer JSON config
        Returns:
            a batch of best answers
        """

        batch_best_answers = []
        batch_context_indices = []
        for contexts, questions in zip(contexts_batch, questions_batch):
            results = []
            for i in range(0, len(contexts), self.batch_size):
                c_batch = contexts[i: i + self.batch_size]
                q_batch = questions[i: i + self.batch_size]
                batch_predict = zip(*self.squad_model(c_batch, q_batch))
                results += batch_predict
            if self.filter_noans:
                results = filter(lambda x: x[0] != '', results)
            sorted_items = sorted(enumerate(results), key=lambda x: x[1][2], reverse=True)[:self.top_n]
            best_answers = list(map(itemgetter(1), sorted_items))
            context_indices = list(map(itemgetter(0), sorted_items))
            best_answers = [ba[::2] for ba in best_answers]
            best_answers = [('no answer', ba[1]) if ba[0] == '' else ba for ba in best_answers]
            if not best_answers:
                best_answers.append([('no answer', -1)])
            batch_best_answers.append(best_answers)
            batch_context_indices.append(context_indices)
        return batch_best_answers, batch_context_indices


@register("tables_logit_ranker")
class TablesLogitRanker(Component):
    """Select best answer using squad model logits. Make several batches for a single batch, send each batch
     to the squad model separately and get a single best answer for each batch.
     Args:
        squad_model: a loaded squad model
        batch_size: batch size to use with squad model
     Attributes:
        squad_model: a loaded squad model
        batch_size: batch size to use with squad model
    """

    def __init__(self, top_n: 3, squad_model: Union[Chainer, Component], batch_size: int = 50,
                 sort_noans: bool = False, **kwargs):
        self.squad_model = squad_model
        self.batch_size = batch_size
        self.top_n = top_n
        self.sort_noans = sort_noans

    def __call__(self, contexts_batch: List[List[str]], questions_batch: List[str], cells_batch,
                 batch_table_indices):
        """
        Sort obtained results from squad reader by logits and get the answer with a maximum logit.
        Args:
            contexts_batch: a batch of contexts which should be treated as a single batch in the outer JSON config
            questions_batch: a batch of questions which should be treated as a single batch in the outer JSON config
        Returns:
            a batch of best answers
        """

        # Get SQuAD answers

        batch_best_answers = []
        for instance_contexts, instance_cells, instance_table_ids, question in zip(contexts_batch, cells_batch,
                                                                                   batch_table_indices,
                                                                                   questions_batch):
            results = []
            for contexts, cells, table_id in zip(instance_contexts, instance_cells, instance_table_ids):
                # if any(item == '' for item in cells):
                #     print('emptu cell!')
                questions = [question] * len(contexts)
                # print(question)
                # print(table_id)
                for i in range(0, len(contexts), self.batch_size):
                    c_batch = contexts[i: i + self.batch_size]
                    q_batch = questions[i: i + self.batch_size]
                    squad_batch_predict = zip(*self.squad_model(c_batch, q_batch))
                    # results.append((list(squad_batch_predict), table_id, question))
                    results.append(
                        [(el[0], el[1], el[2], table_id, cells[i]) for i, el in enumerate(squad_batch_predict)])
            batch_best_answers.append(list(chain.from_iterable(results)))

        # Sorting SQuAD answers

        for k, instance_best_answers in enumerate(batch_best_answers):
            sorted_instance_best_answers = list(filter(lambda x: x[4] != '', instance_best_answers))
            sorted_instance_best_answers = sorted(sorted_instance_best_answers, key=itemgetter(2), reverse=True)[
                                           :self.top_n]
            sorted_instance_best_answers = [(el[4], el[2], el[3]) for el in sorted_instance_best_answers]
            batch_best_answers[k] = sorted_instance_best_answers

        return batch_best_answers
