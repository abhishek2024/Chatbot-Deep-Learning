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


class GoogleDialogsDatasetReader(DatasetReader):

    url = 'http://files.deeppavlov.ai/datasets/google_simulated_dialogues.tar.gz'

    @staticmethod
    @abstractmethod
    def _data_fname(datatype):
        pass

    @classmethod
    @overrides
    def read(self, data_path: str, mode: str = "basic") -> Dict[str, List]:
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
        if mode == 'evaluate':
            data = {
                'train': [],
                'valid': [],
                'test': data['train'] + data['valid'] + data['test']
            }
        elif mode == 'full_train':
            data['train'] = data['train'] + data['test']
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

    @classmethod
    def _get_turns(cls, dialogs: List[Dict]) -> List[Tuple[Dict, Dict]]:
        utterances = []
        responses = []
        for dialog in dialogs:
            u, r = {}, {}
            for i, turn in enumerate(dialog['turns']):
                # Format response
                if 'system_acts' in turn:
                    str_acts = sorted(set(map(cls._format_act, turn['system_acts'])))
                    slots = cls._get_bio_markup(turn['system_utterance']['tokens'],
                                                turn['system_utterance']['slots'])
                    # slot_state = {i['slot']: i['value'] for i in turn['dialogue_state']}
                    acts = cls._format_acts2(turn['system_acts'])
                    r = {'text': turn['system_utterance']['text'],
                         'slots': slots,
                         'acts': acts,
                         'act': '+'.join(str_acts)}
                    if 'slot_values' in turn:
                        r['slot_values'] = turn['slot_values']
                # If u and r are nonnull, add them to overall list of turns
                if u and r:
                    utterances.append(u)
                    responses.append(r)
                    u, r = {}, {}
                # Format utterance
                goals = {item['slot']: item['value'] for item in turn['dialogue_state']}
                intents = cls._format_acts2(turn['user_acts'])
                slots = cls._get_bio_markup(turn['user_utterance']['tokens'],
                                            turn['user_utterance']['slots'])
                u = {'text': turn['user_utterance']['text'],
                     'intents': intents,
                     'goals': goals,
                     'slots': slots}
                if 'slot_values' in turn:
                    u['slot_values'] = turn['slot_values']
                if i == 0:
                    u['episode_done'] = True

            utterances.append(u)
            responses.append(r)

        return utterances, responses

    @staticmethod
    def _format_act(act):
        act_tokens = [act['type'].lower()]
        if 'slot' in act:
            act_tokens.append(act['slot'])
        return '_'.join(act_tokens)

    @staticmethod
    def _format_acts(acts, state):
        new_acts = []
        for a in acts:
            new_acts.append({'act': a['type'].lower(), 'slots': []})
            if 'slot' in a:
                if a['slot'] in state:
                    new_acts[-1]['slots'] = [(a['slot'], state[a['slot']])]
                else:
                    new_acts[-1]['slots'] = [('slot', a['slot'])]
        return new_acts

    @staticmethod
    def _format_acts2(acts):
        new_acts = {}
        for a in acts:
            slot = a['type'].lower()
            if 'slot' in a:
                if a['slot'] not in new_acts.get(slot, []):
                    new_acts[slot] = new_acts.get(slot, []) + [a['slot']]
            elif slot not in new_acts:
                new_acts[slot] = new_acts.get(slot, [])
        return [{'act': a, 'slots': v} for a, v in new_acts.items()]

    @staticmethod
    def _get_bio_markup(tokens, slots):
        markup = ['O'] * len(tokens)
        for s in slots:
            if any(m != 'O' for m in markup[s['start']:s['exclusive_end']]):
                raise RuntimeError('Mentions of different slots shouldn\'t intersect',
                                   tokens, markup, s)
            markup[s['start']] = 'B-{}'.format(s['slot'])
            for i in range(s['start'] + 1, s['exclusive_end']):
                markup[i] = 'I-{}'.format(s['slot'])
        return markup


@register('sim_r_reader')
class GoogleSimRDatasetReader(GoogleDialogsDatasetReader):

    @staticmethod
    @overrides
    def _data_fname(datatype):
        assert datatype in ('train', 'dev', 'test'), "wrong datatype name"
        return 'sim-R/{}.json'.format(datatype)


@register('sim_m_reader')
class GoogleSimMDatasetReader(GoogleDialogsDatasetReader):

    @staticmethod
    @overrides
    def _data_fname(datatype):
        assert datatype in ('train', 'dev', 'test'), "wrong datatype name"
        return 'sim-M/{}.json'.format(datatype)


@register('dstc2_google_reader')
class GoogleDSTC2DatasetReader(GoogleDialogsDatasetReader):

    @staticmethod
    @overrides
    def _data_fname(datatype):
        assert datatype in ('train', 'dev', 'test'), "wrong datatype name"
        return 'DSTC2/{}.json'.format(datatype)
