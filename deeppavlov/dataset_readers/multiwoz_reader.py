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
from typing import Dict, List

from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download_decompress, mark_done
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


@register('multiwoz_reader')
class MultiWOZDatasetReader(DatasetReader):
    """
    # TODO: add docs
    """

    url = 'http://files.deeppavlov.ai/datasets/kvret_public.tar.gz'
    url = None

    DATA_FILES = ['data.json', 'dialogue_acts.json', 'valListFile.json',
                  'testListFile.json']

    @classmethod
    @overrides
    def read(self, data_path: str, dialogs: bool = False) -> Dict[str, List]:
        """
        Downloads ``'kvrest_public.tar.gz'``, decompresses, saves files to ``data_path``.

        Parameters:
            data_path: path to save data
            dialogs: flag indices whether to output list of turns or list of dialogs

        Returns:
            dictionary with ``'train'``, ``'valid'`` and ``'test'``.
            Each fields is a list of tuples ``(x_i, y_i)``.
        """
        if dialogs:
            raise NotImplementedError("dialogs=True is not implemented")

        if not all(Path(data_path, f).exists() for f in self.DATA_FILES):
            log.info('[downloading dstc2 from {} to {}]'.format(self.url, data_path))
            download_decompress(self.url, data_path)
            mark_done(data_path)

        data_file = Path(data_path, 'data.json')
        valid_ids = self._read_indices(Path(data_path, 'valListFile.json'))
        test_ids = self._read_indices(Path(data_path, 'testListFile.json'))
        data = {
            'train': self._read_dialogues(data_file, dialogs=dialogs,
                                          exclude_ids=valid_ids + test_ids),
            'valid': self._read_dialogues(data_file, dialogs=dialogs, ids=valid_ids),
            'test':  self._read_dialogues(data_file, dialogs=dialogs, ids=test_ids)
        }
        return data

    @staticmethod
    def _read_indices(file_path):
        return [ln.strip() for ln in open(file_path, 'rt')]

    @classmethod
    def _read_dialogues(cls, file_path, dialogs=False, ids=None, exclude_ids=[]):
        """Returns data from single file"""
        log.info("[loading dialogs from {}]".format(file_path))

        utterances, responses, dialog_indices =\
            cls._get_turns(cls._iter_file(file_path, ids=ids, exclude_ids=exclude_ids),
                           with_indices=True)

        data = list(map(cls._format_turn, zip(utterances, responses)))

        if dialogs:
            return [data[idx['start']:idx['end']] for idx in dialog_indices]
        return data

    @staticmethod
    def _format_turn(turn):
        x = {'text': turn[0]['text'],
             'dialog_id': turn[0]['dialog_id']}
        if turn[0].get('episode_done') is not None:
            x['episode_done'] = turn[0]['episode_done']
        y = {'text': turn[1]['text']}
        return (x, y)

    @staticmethod
    def _check_dialog(dialog):
        expecting_user = True
        for turn in dialog:
            is_user = bool(turn['metadata'] == {})
            if expecting_user and not is_user:
                log.debug(f"Turn is expected to be by user, but"
                          f" it contains metadata = {turn['metadata']}")
                return False
            elif not expecting_user and is_user:
                log.debug(f"Turn is expected to be by system, but"
                          f" it contains empty metadata")
                return False
            expecting_user = not expecting_user
        if not expecting_user:
            log.debug("Last turn is expected to be by system")
            return False
        return True

    @classmethod
    def _iter_file(cls, file_path, ids=None, exclude_ids=[]):
        with open(file_path, 'rt', encoding='utf8') as f:
            data = json.load(f)
        for dialog_id, sample in data.items():
            if (dialog_id in exclude_ids) or (ids is not None and dialog_id not in ids):
                continue
            domains = [d for d, val in sample['goal'].items()
                       if (val != {}) and (val not in ['topic', 'message'])]
            dialog = sample['log']
            if cls._check_dialog(dialog):
                yield dialog_id, dialog, domains
            else:
                log.warn("Skipping dialogue with uuid={dialog_id}: wrong format.")

    @staticmethod
    def _get_turns(data, with_indices=False):
        utterances, responses, dialog_indices = [], [], []
        for dialog_id, dialog, domains in data:
            for i, turn in enumerate(dialog):
                replica = {'text': turn['text']}
                if i == 0:
                    replica['episode_done'] = True
                is_user = bool(turn['metadata'] == {})
                if is_user:
                    replica['dialog_id'] = dialog_id
                    utterances.append(replica)
                else:
                    replica['domain'] = domains
                    responses.append(replica)

            # if last replica was by driver
            if len(responses) != len(utterances):
                utterances[-1]['end_dialogue'] = False
                responses.append({'text': '', 'end_dialogue': True})

            last_utter = responses[-1]['text']
            if last_utter and not last_utter[-1].isspace():
                last_utter += ' '
            responses[-1]['text'] = last_utter + 'END_OF_DIALOGUE'

            dialog_indices.append({
                'start': len(utterances),
                'end': len(utterances) + len(dialog),
            })

        if with_indices:
            return utterances, responses, dialog_indices
        return utterances, responses
