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
import itertools

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator


@register('google_dial_ner_iterator')
class GoogleDialogsNerDatasetIterator(DataLearningIterator):
    """
    Iterates over user utterances for Google Dialogs NER task.
    Dataset takes a dict with fields 'train', 'test', 'valid'.
    A list of pairs (utterance tokens, ner bio tags) is stored in each field.
    """
    @overrides
    def split(self, *args, **kwargs):
        self.train = self._preprocess(self.train)
        self.valid = self._preprocess(self.valid)
        self.test = self._preprocess(self.test)

    @classmethod
    def _preprocess(cls, turns):
        utter_slots = {}
        for turn in turns:
            for utter, slots in cls._get_slots(turn):
                # remove duplicate pairs (utterance, slots)
                if slots not in utter_slots.get(utter, []):
                    utter_slots[utter] = utter_slots.get(utter, []) + [slots]
        return [(u.split(), s) for u in utter_slots.keys() for s in utter_slots[u]]

    @staticmethod
    def _get_slots(turn):
        x, y = turn
        # for utter in [x, y]:
        for utter in [x]:
            yield (utter['text'], utter['slots'])
