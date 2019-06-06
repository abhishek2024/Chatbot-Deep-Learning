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
import itertools
import random
import json
from pathlib import Path
from typing import Dict, List, Tuple

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader


@register('squad_ranking_reader')
class SquadRankingReader(DatasetReader):
    """The class to read dataset for ranking or paraphrase identification with Siamese networks."""

    def read(self, data_path: str,
             num_candidates=10,
             positive_samples=False, **kwargs) -> Dict[str, List[Tuple[List[str], int]]]:
        """Read the dataset for ranking or paraphrase identification with Siamese networks.

        Args:
            data_path: A path to a folder with dataset files.
        """
        self.num_candidates = num_candidates
        self.positive_samples = positive_samples
        dataset = {'train': None, 'valid': None, 'test': None}
        data_path = expand_path(data_path)
        train_fname = data_path / 'train.jsonl'
        valid_fname = data_path / 'dev.jsonl'
        self.contexts_dict = self._build_contexts_dict(valid_fname)
        dataset["train"] = self._preprocess_data_train(train_fname)
        dataset["valid"] = self._preprocess_data_valid_test(valid_fname)
        dataset["test"] = dataset["valid"]
        print(len(dataset['train']))
        print(len(dataset['valid']))
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
        if self.positive_samples:
            data = [el[0] for el in data if el[1] == 1]
            data = list(zip(data, range(len(data))))
        return data

    def _generate_samples(self, i, num_samples):
        contexts = [v for k, v in self.contexts_dict.items() if k != i]
        contexts = list(itertools.chain.from_iterable(contexts))
        random.shuffle(contexts)
        samples = random.sample(contexts, num_samples)
        return samples

    def _build_contexts_dict(self, fname):
        contexts_dict = {}
        with open(fname, 'r') as f:
            for i, el in enumerate(f):
                sample = json.loads(el)
                contexts = [x["context"].replace('\n\n', ' ').replace('\n', ' ').replace('\t', ' ')
                            for x in sample["contexts"]]
                contexts_dict[i] = contexts
        return contexts_dict

    def _preprocess_data_valid_test(self, fname: Path) -> List[Tuple[List[str], int]]:
        data = []
        with open(fname, 'r') as f:
            for i, el in enumerate(f):
                sample = json.loads(el)
                query = sample["question"]
                contexts = [x["context"].replace('\n\n', ' ').replace('\n', ' ').replace('\t', ' ')
                            for x in sample["contexts"]]
                labels = [1 if len(x["answer"]) != 0 else 0 for x in sample["contexts"]]
                pc = [el[0] for el in zip(contexts, labels) if el[1] == 1]
                nc = [el[0] for el in zip(contexts, labels) if el[1] == 0]
                for c in pc:
                    context_response = [query] + [c] + nc
                    context_response = context_response[:self.num_candidates+1]
                    s_lengh_to_add = self.num_candidates + 1 - len(context_response)
                    if s_lengh_to_add > 0:
                        add_samples = self._generate_samples(i, s_lengh_to_add)
                        context_response += add_samples
                    assert len(context_response) == self.num_candidates + 1
                    data.append((context_response, 1))
        return data
