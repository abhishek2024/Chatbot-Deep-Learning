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


@register('google_dialog_nlu_iterator')
class GoogleDialogsNLUDatasetIterator(DataLearningIterator):
    """
    Iterates over user utterances for Google Dialogs NLU task.
    Dataset takes a dict with fields 'train', 'test', 'valid'.
    A list of pairs (utterance tokens, (ner bio tags, list of intents)) is stored
    in each field.
    """
    @overrides
    def split(self, *args, **kwargs):
        self.train = self._preprocess(self.train)
        self.valid = self._preprocess(self.valid)
        self.test = self._preprocess(self.test)

    @classmethod
    def _preprocess(cls, turns):
        utter_slots = {}
        utter_acts = {}
        for turn in turns:
            for utter, slots, acts in cls._iter_samples(turn):
                # remove duplicate pairs (utterance, slots)
                if slots not in utter_slots.get(utter, []):
                    utter_slots[utter] = utter_slots.get(utter, []) + [slots]
                    utter_acts[utter] = acts
        return [(u.split(), (s, utter_acts[u]))
                for u in utter_slots.keys() for s in utter_slots[u]]

    @classmethod
    def _iter_samples(cls, turn):
        x, y = turn
        # for utter in [x, y]:
        for utter in [x]:
            acts = utter['intents'] if 'intents' in utter else utter['acts']
            yield (utter['text'], utter['slots'], cls._format_acts(acts))

    @staticmethod
    def _format_acts(acts):
        return [a['act'] for a in acts]
