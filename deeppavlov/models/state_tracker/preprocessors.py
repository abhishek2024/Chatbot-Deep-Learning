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

from typing import List, Dict, Any, Union, Tuple
from itertools import chain
import numpy as np

from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.simple_vocab import SimpleVocabulary
from deeppavlov.core.common.log import get_logger
from deeppavlov.dataset_iterators.dialog_iterator import DialogStateDatasetIterator

log = get_logger(__name__)


def tag2slot(tag):
    if tag != 'O':
        if (not tag.startswith('I-')) and (not tag.startswith('B-')):
            raise RuntimeError(f"Wrong tag format: {tag}")
        return tag[2:]


class Delexicalizor(Component):
    """Given an utterance replaces mentions of slots with #slot"""
    def __init__(self, **kwargs):
        pass

    def __call__(self, utterances: List[List[str]], tags: List[List[str]])\
            -> List[List[str]]:
        norm_utterances = []
        for utt_words, utt_tags in zip(utterances, tags):
            utt_norm = [w if t == 'O' else '#' + tag2slot(t)
                        for w, t in zip(utt_words, utt_tags)]
            norm_utterances.append(utt_norm)
        return norm_utterances


class SlotsTokensMatrixBuilder(Component):
    """
    Builds one-hot encoded slot value matrix of shape [num_slots, num_tokens].
    Inputs bio-markuped utterance and vocabulary of slot names.
    """
    def __init__(self, slot_vocab, **kwargs):
        log.info("Found vocabulary with the following slot names: {}"
                 .format(list(slot_vocab.keys())))
        self.slot_vocab = slot_vocab

    def _slot2idx(self, slot):
        if slot not in self.slot_vocab:
            return 0
            #raise RuntimeError(f"Utterance slot {slot} doesn't match any slot"
            #                   " from slot_vocab.")
        #print(f"slot '{slot}' has idx =", list(self.slot_vocab.keys()).index(slot))
        #print(f"returning idx =", self.slot_vocab([[slot]])[0][0])
        return self.slot_vocab([[slot]])[0][0]

    def __call__(self, utterances: List[List[str]], tags: List[List[str]],
                 candidates: List[Dict[str, List[str]]] = [[]]) -> List[np.ndarray]:
        # dirty hack to fix that everything must be a batch
        if len(candidates) != 1:
            raise NotImplementedError("not implemented for candidates with length > 1")
        candidates = candidates[0]
        utt_matrices = []
        for utt, utt_tags in zip(utterances, tags):
            mat = np.zeros((len(self.slot_vocab), len(utt_tags)), dtype=int)
            i = 0
            while (i < len(utt_tags)):
                slot = tag2slot(utt_tags[i])
                if slot is not None:
                    slot_len = 1
                    while (i + slot_len < len(utt_tags)) and\
                            (utt_tags[i + slot_len] == 'I-' + slot):
                        slot_len += 1
                    if candidates is not None:
                        # resulting matrix contains indeces of slot values
                        slot_value = ' '.join(utt[i:i + slot_len])
                        if slot not in candidates:
                            raise RuntimeError(f"slot `{slot}` is not in candidates")
                        if slot_value not in candidates[slot]:
                            raise RuntimeError(f"value `{slot_value}` of slot `{slot}`"
                                               " is not in candidates")
                        slot_value_idx = candidates[slot].index(slot_value)
                        mat[self._slot2idx(slot), i: i + slot_len] = slot_value_idx + 1
                    else:
                        # resulting matrix is a mask of slots
                        mat[self._slot2idx(slot), i] = 1
                    i += slot_len
                else:
                    i += 1
            utt_matrices.append(mat)
        return utt_matrices


class SlotsValuesMatrixBuilder(Component):
    """
    Builds matrix with slot values' scores of shape [num_slots, max_num_values].
    Inputs slot values with scores, if score is missing, a score equal to 1.0 is set.
    """
    def __init__(self, slot_vocab: callable, max_num_values: int, **kwargs) -> None:
        self.slot_vocab = slot_vocab
        self.max_num_values = max_num_values

    def _slot2idx(self, slot):
        if slot not in self.slot_vocab:
            raise RuntimeError(f"Utterance slot '{slot}' doesn't match any slot"
                               " from slot_vocab.")
        return self.slot_vocab([[slot]])[0][0]

    @staticmethod
    def _value2idx(slot, value, candidates):
        if value not in candidates[slot]:
            # log.info(f"slot's '{slot}' value '{value}' doesn't match"
            #         f" any value from candidates {candidates}.")
            # raise RuntimeError(f"'{slot}' slot's value '{value}' doesn't match"
            #                   " any value from candidates {candidates}.")
            return 0
        return candidates[slot].index(value)

    @staticmethod
    def _format_slot_dict(x):
        if isinstance(x, Dict):
            return ({'slot': k, 'value': v} for k, v in x.items())
        return x

    def __call__(self, slots: Union[List[Dict[str, Any]], Dict[str, Any]],
                 candidates: List[Dict[str, List[str]]]) -> List[np.ndarray]:
        # dirty hack to fix that everything must be a batch
        if len(candidates) != 1:
            raise NotImplementedError("not implemented for candidates with length > 1")
        candidates = candidates[0]
        utt_matrices = []
        for utt_slots in map(self._format_slot_dict, slots):
            mat = np.zeros((len(self.slot_vocab), self.max_num_values + 2), dtype=float)
            for s in utt_slots:
                slot_idx = self._slot2idx(s['slot'])
                value_idx = self._value2idx(s['slot'], s['value'], candidates)
                mat[slot_idx, value_idx] = s.get('score', 1.)
            utt_matrices.append(mat)
        return utt_matrices


class SlotsActionsMatrixBuilder(Component):
    """
    Builds matrics with mask of actions and corresponding slots of shape
    [num_slots, num_actions].
    Inputs actions as a list of dicts in the format
    {'action': act, 'slots': [slot1, slot2]}.
    """
    def __init__(self, slot_vocab: callable, action_vocab: callable, **kwargs) -> None:
        self.slot_vocab = slot_vocab
        self.action_vocab = action_vocab

    def __call__(self, actions: List[List[Dict]]) -> List[np.ndarray]:
        utt_matrices = []
        for utt_acts in actions:
            mat = np.zeros((len(self.slot_vocab), len(self.action_vocab)), dtype=float)
            for a in utt_acts:
                act_idx = self.action_vocab[a['act']]
                slot_idxs = self.slot_vocab([a.get('slots', [])])[0]
                mat[slot_idxs, act_idx] = 1.
            utt_matrices.append(mat)
        return utt_matrices


class SlotsBioMarkup2Dict(Component):
    """
    Converts batch of tuples (utterance tokens, slot tags) to dictionary
    with slots as keys and slot mentions as values.
    Tags can be of three types: 'O' -- out of slot, 'B-slot' -- begin of slot 'slot',
    'I-slot' -- inside of slot 'slot'.
    """
    def __init__(self, **kwargs) -> None:
        pass

    def __call__(self, utter: Union[List[str], List[List[str]]],
                 tags: Union[List[str], List[List[str]]]) -> List[Dict[str, str]]:
        if isinstance(utter[0], str):
            return [DialogStateDatasetIterator._biomarkup2dict(utter, tags)]
        return [DialogStateDatasetIterator._biomarkup2dict(u, t)
                for u, t in zip(utter, tags)]


class SlotsValuesMatrix2Dict(Component):
    """
    Converts matrix of slot values of shape [num_slots, max_num_values]
    to dictionary with slots as keys and values as dictionary values.
    """
    def __init__(self, slot_vocab: callable, exclude_values: List = [], **kwargs) -> None:
        self.slot_vocab = slot_vocab
        self.exclude_values = exclude_values

    @staticmethod
    def _idx2value(slot, idx, candidates):
        idx = idx if idx < len(candidates[slot]) else 0
        return candidates[slot][idx]

    def __call__(self, slots_values: np.ndarray,
                 candidates: List[Dict[str, List[str]]]) -> List[Dict[str, str]]:
        # dirty hack to fix that everything must be a batch
        if len(candidates) != 1:
            raise NotImplementedError("not implemented for candidates with length > 1")
        candidates = candidates[0]
        slot_dict = {}
        # for slot_idx, value_idx in zip(*np.where(slots)):
        for slot_idx, value_idx in enumerate(slots_values):
            slot = self.slot_vocab([[slot_idx]])[0][0]
            value = self._idx2value(slot, value_idx, candidates)
            if value not in self.exclude_values:
                slot_dict[slot] = value
        return [slot_dict]


class ActionSlotsMatcher(Component):
    """
    Matches slots with actions.
    Inputs list of slots and list of string actions from NLU model.
    Outputs list of actions matched with slots.
    """

    def __init__(self, matched_actions: List[str] = [], **kwargs) -> None:
        self.matched_actions = matched_actions

    def _get_priority(self, action):
        if action in self.matched_actions:
            return len(self.matched_actions) - self.matched_actions.index(action)
        return 0

    def __call__(self, actions: Union[List[List[str]], List[List[Dict[str, Any]]]],
                 slots: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        # dirty hack to fix that everything must be a batch
        if len(slots) != 1:
            raise NotImplementedError("not implemented for slots with length > 1")
        slot_names = [s['slot'] for s in slots[0]]
        new_actions = []
        for a_list in actions:
            new_actions.append([])
            matched = False
            if a_list and isinstance(a_list[0], dict):
                a_list = [a['act'] for a in a_list]
            for a in sorted(a_list, key=self._get_priority, reverse=True):
                if a in self.matched_actions:
                    if not matched:
                        new_actions[-1].append({'act': a, 'slots': slot_names})
                        matched = True
                    else:
                        new_actions[-1].append({'act': a, 'slots': []})
                        # print(f"Couldn't match slots {slots} with"
                        #      f" actions {actions}.")
                else:
                    new_actions[-1].append({'act': a, 'slots': []})
        return new_actions


@register('action_vocab')
class ActionVocabulary(SimpleVocabulary):
    """Implements action vocabulary."""

    @staticmethod
    def _format_action(a):
        if isinstance(a, Dict):
            return a['act']
        return a

    def fit(self, *args):
        actions = [self._format_action(a) for a in chain(*chain(*args))]
        super().fit([actions])

    def __call__(self, batch: List[List[Union[Dict, int]]], **kwargs) ->\
            List[List[Union[Dict, int]]]:
        if isinstance(batch, (str, int)):
            return super().__call__(batch)
        return super().__call__([self._format_action(a) for a in chain(*batch)])
