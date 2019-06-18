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
from pathlib import Path
from typing import List, Dict, Tuple
import random
from collections import defaultdict

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader


@register('msmarco_reader')
class MSMARCOReader(DatasetReader):
    """The class to read the Ubuntu V2 dataset from csv files.

    Please, see https://github.com/rkadlec/ubuntu-ranking-dataset-creator.
    """

    def read(self, data_path: str,
             positive_samples=False,
             random_seed=243,
             *args, **kwargs) -> Dict[str, List[Tuple[List[str], int]]]:
        """Read the Ubuntu V2 dataset from csv files.

        Args:
            data_path: A path to a folder with dataset csv files.
            positive_samples: if `True`, only positive context-response pairs will be taken for train
        """

        data_path = expand_path(data_path)
        dataset = {'train': None, 'valid': None, 'test': None}
        self.positive_samples = positive_samples
        random.seed(random_seed)
        self.collection_fname = Path(data_path) / 'collection.tsv'
        self.queries_train_fname = Path(data_path) / 'queries.train.tsv'
        self.qrels_train_fname = Path(data_path) / 'qrels.train.tsv'
        self.queries_dev_fname = Path(data_path) / 'queries.dev' \
                                                   '.tsv'
        self.qrels_dev_fname = Path(data_path) / 'qrels.dev.small.tsv'
        self.top_dev_fname = Path(data_path) / 'top1000.dev.tsv'
        self.collection = self.build_collection()
        self.collection_size = len(self.collection)
        dataset["train"] = self.preprocess_data_train()
        dataset["valid"] = self.preprocess_data_validation()
        dataset["test"] = dataset["valid"]
        return dataset

    def build_collection(self):
        collection = dict()
        with open(self.collection_fname) as f:
            reader = csv.reader(f, delimiter='\t')
            for el in reader:
                collection[el[0]] = el[1]
        return collection

    def preprocess_data_train(self) -> List[Tuple[List[str], int]]:
        qid_query = dict()
        with open(self.queries_train_fname) as f:
            reader = csv.reader(f, delimiter='\t')
            for el in reader:
                qid_query[el[0]] = el[1]

        queries = []
        responses = []
        with open(self.qrels_train_fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for el in reader:
                queries.append(qid_query[el[0]])
                responses.append(self.collection[el[2]])
                if not self.positive_samples:
                    pid = str(random.randint(0, self.collection_size))
                    queries.append(qid_query[el[0]])
                    responses.append(self.collection[pid])

        data = list(zip(queries, responses))
        if self.positive_samples:
            labels = range(len(data))
        else:
            labels = (len(data) // 2) * [1, 0]
        data = list(zip(data, labels))
        return data

    def preprocess_data_validation(self) -> List[Tuple[List[str], int]]:
        qid_query = dict()
        with open(self.queries_dev_fname) as f:
            reader = csv.reader(f, delimiter='\t')
            for el in reader:
                qid_query[el[0]] = el[1]

        qid_relevant_pids = defaultdict(list)
        with open(self.qrels_dev_fname) as f:
            reader = csv.reader(f, delimiter='\t')
            for el in reader:
                qid_relevant_pids[el[0]].append(el[2])

        qid_candidate_pids = defaultdict(list)
        with open(self.top_dev_fname, newline='') as f:
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for el in reader:
                if len(el) == 4 and el[0].isdigit() and el[1].isdigit():
                    qid_candidate_pids[el[0]].append(el[1])

        data = []
        labels = []
        for qid in qid_candidate_pids:
            sample = []
            cand_pids = qid_candidate_pids[qid]
            rel_pids = qid_relevant_pids[qid]
            count = 0
            for pid in rel_pids:
                if pid in cand_pids:
                    cand_pids.remove(pid)
                    cand_pids.insert(0, pid)
                    count +=1
            labels.append(count)
            sample.append(qid_query[qid])
            for pid in cand_pids:
                sample.append(self.collection[pid])
            if len(sample) < 1001:
                sample += (1001 - len(sample)) * ['EMPTY_PASSAGE']
            assert(len(sample) == 1001)
            data.append(sample)
        data = list(zip(data, labels))
        data = data[:100]
        # data = random.sample(data, 200)
        return data
