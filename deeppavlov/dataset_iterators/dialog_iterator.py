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

from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator


@register('dialog_iterator')
class DialogDatasetIterator(DataLearningIterator):
    """
    Iterates over dialog data,
    generates batches where one sample is one dialog.

    A subclass of
    :class:`~deeppavlov.core.data.data_learning_iterator.DataLearningIterator`.

    Attributes:
        train: list of training dialogs (tuples ``(context, response)``)
        valid: list of validation dialogs (tuples ``(context, response)``)
        test: list of dialogs used for testing (tuples ``(context, response)``)
    """

    @staticmethod
    def _dialogs(data):
        dialogs = []
        prev_resp_act = None
        for x, y in data:
            if x.get('episode_done'):
                del x['episode_done']
                prev_resp_act = None
                dialogs.append(([], []))
            x['prev_resp_act'] = prev_resp_act
            prev_resp_act = y['act']
            dialogs[-1][0].append(x)
            dialogs[-1][1].append(y)
        return dialogs

    @overrides
    def split(self, *args, **kwargs):
        self.train = self._dialogs(self.train)
        self.valid = self._dialogs(self.valid)
        self.test = self._dialogs(self.test)


@register('dialog_db_result_iterator')
class DialogDBResultDatasetIterator(DataLearningIterator):
    """
    Iterates over dialog data,
    outputs list of all ``'db_result'`` fields (if present).

    The class helps to build a list of all ``'db_result'`` values present in a dataset.

    Inherits key methods and attributes from
    :class:`~deeppavlov.core.data.data_learning_iterator.DataLearningIterator`.

    Attributes:
        train: list of tuples ``(db_result dictionary, '')`` from "train" data
        valid: list of tuples ``(db_result dictionary, '')`` from "valid" data
        test: list of tuples ``(db_result dictionary, '')`` from "test" data
    """
    @staticmethod
    def _db_result(data):
        x, y = data
        if 'db_result' in x:
            return x['db_result']

    @overrides
    def split(self, *args, **kwargs):
        self.train = [(r, "") for r in filter(None, map(self._db_result, self.train))]
        self.valid = [(r, "") for r in filter(None, map(self._db_result, self.valid))]
        self.test = [(r, "") for r in filter(None, map(self._db_result, self.test))]


@register('dialog_state_iterator')
class DialogStateDatasetIterator(DataLearningIterator):
    """
    Iterates over dialog data,
    outputs list of pairs ((utterance tokens, user slots, previous system slots,
    previous dialog state), (dialog state)).

    Inherits key methods and attributes from
    :class:`~deeppavlov.core.data.data_learning_iterator.DataLearningIterator`.

    Attributes:
        train:
        valid:
        test:
    """
    @overrides
    def split(self, *args, **kwargs):
        self.train = self._preprocess(self.train)
        self.valid = self._preprocess(self.valid)
        self.test = self._preprocess(self.test)

    @classmethod
    def _preprocess(cls, turns):
        data = []
        prev_turn = {}
        for u, r in turns:
            if u.get('episode_done'):
                prev_turn = {}
            u_tokens = u['text'].split()
            r_tokens = r['text'].split() if 'text' in r else []
            u_slots = cls._biomarkup2dict(u_tokens, u['slots'])
            r_slots = cls._biomarkup2dict(r_tokens, r.get('slots', []))

            x_tuple = (u_tokens, u_slots, u['slots'],
                       prev_turn.get('system_tokens', []),
                       prev_turn.get('system_slots', {}),
                       prev_turn.get('system_slots_bio', []),
                       prev_turn.get('goals', {}))
            y_tuple = (u['goals'])
            prev_turn = {'goals': u['goals'], 'system_tokens': r_tokens,
                         'system_slots': r_slots, 'system_slots_bio': r.get('slots', [])}

            data.append((x_tuple, y_tuple))
        return data

    @staticmethod
    def _biomarkup2list(tokens, markup):
        slots = []
        _markup = markup + ['O']
        i = 0
        while i < len(_markup):
            if _markup[i] != 'O':
                slot = _markup[i][2:]
                slot_exclusive_end = i + 1
                while _markup[slot_exclusive_end] == ('I-' + slot):
                    slot_exclusive_end += 1
                slots.append((slot, ' '.join(tokens[i: slot_exclusive_end])))
                i = slot_exclusive_end
            else:
                i += 1
        return slots

    @staticmethod
    def _biomarkup2dict(tokens, markup):
        slots = []
        _markup = markup + ['O']
        i = 0
        while i < len(_markup):
            if _markup[i] != 'O':
                slot = _markup[i][2:]
                slot_exclusive_end = i + 1
                while _markup[slot_exclusive_end] == ('I-' + slot):
                    slot_exclusive_end += 1
                slots.append({'slot': slot,
                              'value': ' '.join(tokens[i: slot_exclusive_end])})
                i = slot_exclusive_end
            else:
                i += 1
        return slots
