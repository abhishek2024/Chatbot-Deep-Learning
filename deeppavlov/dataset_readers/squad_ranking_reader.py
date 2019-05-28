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

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader


@register('squad_ranking_reader')
class SquadRankingReader(DatasetReader):
    """The class to read dataset for ranking or paraphrase identification with Siamese networks."""

    def read(self, data_path: str, **kwargs) -> Dict[str, List[Tuple[List[str], int]]]:
        """Read the dataset for ranking or paraphrase identification with Siamese networks.

        Args:
            data_path: A path to a folder with dataset files.
        """

        dataset = {'train': None, 'valid': None, 'test': None}
        data_path = expand_path(data_path)
        train_fname = data_path / 'train.jsonl'
        valid_fname = data_path / 'dev.jsonl'
        dataset["train"] = self._preprocess_data_train(train_fname)
        dataset["valid"] = self._preprocess_data_valid_test(valid_fname)
        dataset["test"] = dataset["valid"]
        return dataset

    def _preprocess_data_train(self, fname: Path) -> List[Tuple[List[str], int]]:
        data = []
        with open(fname, 'r') as f:
            for el in f:
                sample = json.loads(el)
                query = sample["question"]
                contexts = [x["context"].replace('\n\n', ' ').replace('\n', ' ').replace('\t', ' ') for x in sample["contexts"]]
                labels = [1 if len(x["answer"]) != 0 else 0 for x in sample["contexts"]]
                for c, l in zip(contexts, labels):
                    data.append(([query, c], l))
        return data

    def _preprocess_data_valid_test(self, fname: Path) -> List[Tuple[List[str], int]]:
        data = []
        with open(fname, 'r') as f:
            for el in f:
                sample = json.loads(el)
                query = sample["question"]
                contexts = [x["context"].replace('\n\n', ' ').replace('\n', ' ').replace('\t', ' ') for x in sample["contexts"]]
                labels = [1 if len(x["answer"]) != 0 else 0 for x in sample["contexts"]]
                pos_conts = [el[0] for el in zip(contexts, labels) if el[1] == 1]
                neg_conts = [el[0] for el in zip(contexts, labels) if el[1] == 0]
                for c in pos_conts:
                    sample = [query] + [c] + neg_conts
                    sample += sample + (22 - len(sample)) * ["EMPTY_DOCUMENT"]
                    data.append((sample, 1))
        return data
