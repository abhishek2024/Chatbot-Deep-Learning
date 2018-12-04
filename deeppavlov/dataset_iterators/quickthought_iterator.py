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
from random import Random
from copy import copy

from typing import List, Tuple, Iterator, Optional, Dict, Any
from  collections import deque

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.dataset_iterators.file_paths_iterator import FilePathsIterator
from deeppavlov.core.common.log import get_logger

log = get_logger(__name__)

@register('quickthought_iterator')
class QuickThoughtIterator(FilePathsIterator):
    """
    Train: pairs of sent1, sent2 where sent2 follows sent1 in the text.
        Negative samples are made automatically by the model.
    Valid: pairs of (sent1, sent2), bool where bool is the label
        whether sent2 follows sent1 in the text.
    """
    def __init__(self, data: Dict[str, List[Tuple[Any, Any]]], seed: int = None, shuffle: bool = True,
                 *args, **kwargs) -> None:
        self.shuffle = shuffle

        self.valid_seed = seed or 42  # valid set should always be reproducible
        self.random = Random(seed)

        self.train = data.get('train', [])
        self.valid = data.get('valid', [])
        self.test = data.get('test', [])
        self.split(*args, **kwargs)
        self.data = {
            'train': self.train,
            'valid': self.valid,
            'test': self.test,
            'all': self.train + self.test + self.valid
        }

    @staticmethod
    def _read_prepare_data(path):
        """
        Make pairs of sentences sent1, sent2
        where sent2 follows sent1 in the text
        """
        sentences_pairs = []
        last_sentence = None
        with open(path, 'r', encoding="utf-8") as f:
            for i, current_sentence in enumerate(f):
                current_sentence = current_sentence.rstrip('\n')
                if not current_sentence:  # document delimiter
                    last_sentence = None
                    continue

                if last_sentence is None:
                    last_sentence = current_sentence
                else:
                    # i for triplet loss - different i for different pairs
                    current_pair = (last_sentence, current_sentence), i
                    last_sentence = current_sentence
                    sentences_pairs.append(current_pair)
        return sentences_pairs

    def _chunk_generator(self, items_list, chunk_size, data_type):
        random = self.random
        if data_type != 'train':
            # for valid set reproducibility
            random = Random(self.valid_seed)

        chunk = []
        for i in range(len(items_list)):
            (sent1, sent2), label = items_list[i]
            label = 1
            if random.random() < 0.5:
                # items_list item structure: ((sent_1, sent2), label)
                sent2 = random.choice(items_list)[0][1]
                label = 0
            chunk.append(((sent1, sent2), label))

            if len(chunk) == chunk_size:
                chunk_to_yield = copy(chunk)
                chunk = []
                yield chunk_to_yield
        if chunk:
            yield chunk

    def _shard_generator(self, shards, shuffle):
        """
        Params:
            shards: generator of file paths
        """
        shards_to_choose = list(shards)
        if shuffle:
            self.random.shuffle(shards_to_choose)
        for shard in shards_to_choose:
            log.info(f'Loaded shard from {shard}')
            sentences_pairs = self._read_prepare_data(shard)
            if shuffle:
                self.random.shuffle(sentences_pairs)
            yield sentences_pairs 

    def gen_batches(self, batch_size: int, data_type: str = 'train', shuffle: bool = None)\
            -> Iterator[Tuple[str,str]]:
        if shuffle is None:
            shuffle = self.shuffle

        tgt_data = self.data[data_type]
        shard_generator = self._shard_generator(tgt_data, shuffle)

        bs = batch_size
        for shard in shard_generator:
            if not batch_size:
                bs = len(shard)

            lines_generator = self._chunk_generator(shard, bs, data_type)
            for i, lines in enumerate(lines_generator):
                yield tuple(zip(*lines))
