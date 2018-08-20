# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, softwaredata
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
from pathlib import Path
from typing import Dict, List, Tuple
from abc import abstractmethod

from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download_decompress, mark_done
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


class GoogleDatasetReader(DatasetReader):

    url = 'http://files.deeppavlov.ai/datasets/google_simulated_dialogues.tar.gz'

    @staticmethod
    @abstractmethod
    def _data_fname(datatype):
        pass

    @classmethod
    @overrides
    def read(self, data_path: str) -> Dict[str, List]:
        """
        Parameters:
            data_path: path to save data

        Returns:
            dictionary that contains ``'train'`` field with dialogs from
            ``'train.json'``, ``'valid'`` field with dialogs from
            ``'valid.json'`` and ``'test'`` field with dialogs from
            ``'test.json'``. Each field is a list of tuples ``(x_i, y_i)``.
        """
        required_files = (Path(data_path) / self._data_fname(dt)
                          for dt in ('train', 'dev', 'test'))
        if not all(f.exists() for f in required_files):
            log.info('[downloading data from {} to {}]'.format(self.url, data_path))
            download_decompress(self.url, data_path)
            mark_done(data_path)

        data = {
            'train': self._read_from_file(Path(data_path, self._data_fname('train'))),
            'valid': self._read_from_file(Path(data_path, self._data_fname('dev'))),
            'test': self._read_from_file(Path(data_path, self._data_fname('test')))
        }
        return data

    @classmethod
    def _read_from_file(cls, file_path: str) -> List[Tuple[Dict, Dict]]:
        """Returns data from single file"""

        log.info("[loading dialogs from {}]".format(file_path))
        utterances, responses = cls._get_turns(cls._iter_file(file_path))
        return list(zip(utterances, responses))

    @staticmethod
    def _iter_file(file_path):
        for dialogue in json.load(open(file_path, 'rt', encoding='utf8')):
            yield dialogue

    @staticmethod
    def _format_act(act):
        act_tokens = [act['type'].lower()]
        if 'slot' in act:
            act_tokens.append(act['slot'])
        return '_'.join(act_tokens)

    @classmethod
    def _get_turns(cls, dialogs: List[Dict]) -> List[Tuple[Dict, Dict]]:
        utterances = []
        responses = []
        for dialog in dialogs:
            u, r = {}, {}
            for i, turn in enumerate(dialog['turns']):
                # Format response
                if 'system_acts' in turn:
                    acts = sorted(set(map(cls._format_act, turn['system_acts'])))
                    r = {'text': turn['system_utterance']['text'],
                         'acts': acts,
                         'act': '+'.join(acts)}
                # If u and r are nonnull, add them to overall list of turns
                if u and r:
                    utterances.append(u)
                    responses.append(r)
                    u, r = {}, {}
                # Format utterance
                u = {'text': turn['user_utterance']['text'],
                     'intents': turn['user_acts'],
                     'slots': turn['user_utterance']['slots']}
                if i == 0:
                    u['episode_done'] = True

            utterances.append(u)
            responses.append(r)

        return utterances, responses


@register('sim_r_reader')
class GoogleSimRDatasetReader(GoogleDatasetReader):

    @staticmethod
    @overrides
    def _data_fname(datatype):
        assert datatype in ('train', 'dev', 'test'), "wrong datatype name"
        return 'sim-R/{}.json'.format(datatype)


@register('sim_m_reader')
class GoogleSimMDatasetReader(GoogleDatasetReader):

    @staticmethod
    @overrides
    def _data_fname(datatype):
        assert datatype in ('train', 'dev', 'test'), "wrong datatype name"
        return 'sim-M/{}.json'.format(datatype)
