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

import json
import linecache
from copy import copy
from typing import Dict, Iterator, List, Tuple

from sklearn.model_selection import train_test_split

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator

log = get_logger(__name__)


@register("coreference_iterator")
class CorefIterator(DataLearningIterator):
    """Dataset iterator for learning models, e. g. neural networks.

    Args:
        data: list of (x, y) pairs for every data type in ``'train'``, ``'valid'`` and ``'test'``
        seed: random seed for data shuffling
        shuffle: whether to shuffle data during batching

    Attributes:
        shuffle: whether to shuffle data during batching
        random: instance of ``Random`` initialized with a seed
    """

    def __init__(self, data: Dict, mode: str, seed: int = None, shuffle: bool = True, *args, **kwargs) -> None:
        super().__init__(data, seed=seed, shuffle=shuffle, *args, **kwargs)
        self.jsonl_path = None
        self.mode = mode
        if self.mode == "jsonl":
            self.jsonl_path = copy(data['train'])
            with data['train'].open('r', encoding='utf8') as train_file:
                data['train'] = [i for i, line in enumerate(train_file) if line.rstrip()]
        elif self.mode not in ["conll", "jsonl"]:
            raise ConfigError(f"Coreference iterator not supported '{self.mode}' mode.")

        self._split(**kwargs)

    def _split(self, **kwargs):
        if kwargs.get("field_to_split") is not None:
            if kwargs.get("split_fields") is not None:
                log.info("Splitting field <<{}>> to new fields <<{}>>".format(kwargs.get("field_to_split"),
                                                                              kwargs.get("split_fields")))
                self._split_data(field_to_split=kwargs.get("field_to_split"),
                                 split_fields=kwargs.get("split_fields"),
                                 split_proportions=[float(s) for s in
                                                    kwargs.get("split_proportions")],
                                 split_seed=kwargs.get("split_seed"),
                                 stratify=kwargs.get("stratify"))
            else:
                raise IOError("Given field to split BUT not given names of split fields")

    def _split_data(self, field_to_split: str = None, split_fields: List[str] = None,
                    split_proportions: List[float] = None, split_seed: int = None, stratify: bool = None) -> bool:
        """
        Split given field of dataset to the given list of fields with corresponding proportions

        Args:
            field_to_split: field name (out of ``"train", "valid", "test"``) which to split
            split_fields: list of names (out of ``"train", "valid", "test"``) of fields to which split
            split_proportions: corresponding proportions
            split_seed: random seed for splitting dataset
            stratify: whether to use stratified split

        Returns:
            None
        """
        if split_seed is None:
            split_seed = self.random.randint(0, 10000)

        if field_to_split not in ["train", "valid", "test"]:
            raise ConfigError(f"Field to split in dataset iterator config must be 'train', 'valid' or 'test', "
                              f"but '{field_to_split}' was found.")

        data_to_div = self.data[field_to_split].copy()
        data_size = len(self.data[field_to_split])

        for i in range(len(split_fields) - 1):
            if stratify:
                stratify = [sample[1] for sample in data_to_div]
            self.data[split_fields[i]], data_to_div = train_test_split(
                data_to_div,
                test_size=len(data_to_div) - int(data_size * split_proportions[i]),
                random_state=split_seed,
                stratify=stratify)
            self.data[split_fields[-1]] = data_to_div
        return True

    def gen_batches(self, batch_size: int = 1, data_type: str = 'train',
                    shuffle: bool = None) -> Iterator[Tuple[tuple, tuple]]:
        """
        Generate batches of inputs and expected output to train neural networks

        Args:
            batch_size:
            data_type: can be either 'train', 'test', or 'valid'
            shuffle: whether to shuffle dataset before batching

        Yields:
             a tuple of a batch of inputs and a batch of expected outputs
        """
        if shuffle is None:
            shuffle = self.shuffle
        data = self.data[data_type]
        assert len(data) != 0

        if shuffle:
            self.random.shuffle(data)

        if self.mode == 'conll':
            for conll_file in data:
                with conll_file.open("r", encoding='utf8') as cnlf:
                    conll_str = cnlf.read()
                yield tuple(conll_str, )
        else:
            with self.jsonl_path.open('r', encoding='utf8') as json_file:
                for i in data:
                    example = json.loads(linecache.getline(json_file, i))
                    yield example["sentences"], example["speakers"], example["doc_key"], example["clusters"]

    def get_instances(self, data_type: str = 'train') -> Tuple[tuple, tuple]:
        """Get all data for a selected data type

        Args:
            data_type (str): can be either ``'train'``, ``'test'``, ``'valid'`` or ``'all'``

        Returns:
             a tuple of all inputs for a data type and all expected outputs for a data type
        """
        data = []
        if self.mode == 'conll':
            for conll_file in self.data[data_type]:
                with conll_file.open("r", encoding='utf8') as cnlf:
                    conll_str = cnlf.read()
                data.append(conll_str)
        else:
            with self.jsonl_path.open('r', encoding='utf8') as json_file:
                for i in self.data[data_type]:
                    example = json.loads(linecache.getline(json_file, i))
                    data.append((example["sentences"], example["speakers"], example["doc_key"], example["clusters"]))

        return tuple(zip(*data))
