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

from typing import List
import numpy as np

from deeppavlov.core.models.component import Component
from deeppavlov.core.common.log import get_logger

log = get_logger(__name__)


def tag2slot(tag):
    if tag != 'O':
        if (not tag.startswith('I-')) and (not tag.startswith('B-')):
            raise RuntimeError(f"Wrong tag format: {tag}")
        return tag[2:]


class Delexicalizator(Component):
    """Given an utterance replaces mentions of slots with #slot"""
    def __call__(self, utterances: List[List[str]], tags: List[List[str]])\
            -> List[List[str]]:
        norm_utterances = []
        for utt_words, utt_tags in zip(utterances, tags):
            utt_norm = [w if t == 'O' else '#' + tag2slot(t)
                        for w, t in zip(utt_words, utt_tags)]
            norm_utterances.append(utt_norm)
        return norm_utterances


class BioMarkupToMatrix(Component):
    """
    Assembles one-hot encoded slot value matrix of shape [num_slots, num_tokens].
    Inputs bit-markuped utterance and vocabulary of slot names.
    """
    def __init__(self, slot_vocab, **kwargs):
        log.info("Found vocabulary with the following slot names: {}"
                 .format(list(slot_vocab.keys())))
        self.slot_vocab = slot_vocab

    def _slot2idx(self, slot):
        if slot not in self.slot_vocab:
            raise RuntimeError(f"Utterance slot {slot} doesn't match any slot"
                               " from slot_vocab.")
        else:
            print(f"slot '{slot}' has idx =", list(self.slot_vocab.keys()).index(slot))
            print(f"returning idx =", self.slot_vocab([[slot]])[0][0])
            return self.slot_vocab([[slot]])[0][0]

    def __call__(self, tagged_utterances: List[List[str]]) -> List[np.ndarray]:
        utt_matrices = []
        for utt_tags in tagged_utterances:
            mat = np.zeros((len(self.slot_vocab), len(utt_tags)), dtype=int)
            for i, tag in enumerate(utt_tags):
                slot = tag2slot(tag)
                if slot is not None:
                    mat[self._slot2idx(slot), i] = 1
            utt_matrices.append(mat)
        return utt_matrices
