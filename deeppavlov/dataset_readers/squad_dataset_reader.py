"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from pathlib import Path
import json
import pickle

from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download_decompress
from deeppavlov.core.common.registry import register


@register('squad_dataset_reader')
class SquadDatasetReader(DatasetReader):
    """
    Stanford Question Answering Dataset
    https://rajpurkar.github.io/SQuAD-explorer/
    and
    Russian dataset from SDSJ
    https://www.sdsj.ru/ru/contest.html
    """

    url_squad = 'http://files.deeppavlov.ai/datasets/squad-v1.1.tar.gz'
    url_squad_2_0 = 'http://files.deeppavlov.ai/datasets/squad-v2.0.tar.gz'
    url_sber_squad = 'http://files.deeppavlov.ai/datasets/sber_squad-v1.1.tar.gz'
    url_multi_squad = 'http://files.deeppavlov.ai/datasets/multiparagraph_squad.tar.gz'
    url_ner_squad = 'http://files.deeppavlov.ai/datasets/squad-tokens-ner-v1.1.tar.gz'

    def read(self, dir_path: str, dataset='SQuAD'):
        required_files = ['{}-v1.1.json'.format(dt) for dt in ['train', 'dev']]

        if dataset == 'SQuAD':
            self.url = self.url_squad
        elif dataset == 'SQuAD 2.0':
            self.url = self.url_squad_2_0
            required_files = ['{}-v2.0.json'.format(dt) for dt in ['train', 'dev']]
        elif dataset == 'SberSQuAD':
            self.url = self.url_sber_squad
        elif dataset == 'MultiSQuAD':
            self.url = self.url_multi_squad
            required_files = ['{}_multi-v1.1.json'.format(dt) for dt in ['train', 'dev']]
        elif dataset == 'NERSQuAD':
            self.url = self.url_ner_squad
            required_files = ['{}-tokens-ner-v1.1.json'.format(dt) for dt in ['train', 'dev']]
        else:
            raise RuntimeError('Dataset {} is unknown'.format(dataset))

        dir_path = Path(dir_path)
        if not dir_path.exists():
            dir_path.mkdir()

        if not all((dir_path / f).exists() for f in required_files):
            download_decompress(self.url, dir_path)

        dataset = {}
        for f in required_files:
            data = json.load((dir_path / f).open('r'))
            if f in (['dev-v{}.json'.format(v) for v in ['1.1', '2.0']] + ['dev_multi-v1.1.json', 'dev-tokens-ner-v1.1.json']):
                dataset['valid'] = data
            else:
                dataset['train'] = data

        return dataset


@register('squad_scorer_dataset_reader')
class SquadScorerDatasetReader(DatasetReader):
    url = 'http://lnsigo.mipt.ru/export/datasets/selqa_squad_scorer.zip'

    def read(self, dir_path: str):
        dir_path = Path(dir_path)
        required_files = ['selqa_squad_{}.pckl'.format(dt) for dt in ['train', 'dev']]
        if not dir_path.exists():
            dir_path.mkdir()

        if not all((dir_path / f).exists() for f in required_files):
            download_decompress(self.url, dir_path)

        dataset = {}
        for f in required_files:
            data = pickle.load((dir_path / f).open('rb'))
            if f == 'selqa_squad_dev.pckl':
                dataset['valid'] = data
            else:
                dataset['train'] = data

        return dataset


@register('selqa_squad_noans_dataset_reader')
class SquadScorerDatasetReader(DatasetReader):
    url = 'http://lnsigo.mipt.ru/export/datasets/selqa_squad_noans.zip'

    def read(self, dir_path: str):
        dir_path = Path(dir_path)
        required_files = ['selqa_squad_noans_{}.pckl'.format(dt) for dt in ['train', 'dev']]
        if not dir_path.exists():
            dir_path.mkdir()

        if not all((dir_path / f).exists() for f in required_files):
            download_decompress(self.url, dir_path)

        dataset = {}
        for f in required_files:
            data = pickle.load((dir_path / f).open('rb'))
            if f == 'selqa_squad_noans_dev.pckl':
                dataset['valid'] = data
            else:
                dataset['train'] = data

        return dataset
