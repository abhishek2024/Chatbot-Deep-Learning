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
from pathlib import Path
from typing import Union

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader


@register('coref_dialog_reader')
class CorefDialogReader(DatasetReader):
    def read(self, data_path: Union[str, Path], *args, **kwargs):
        if isinstance(data_path, str):
            data_path = Path(data_path)
        with data_path.open("r", encoding='utf8') as dataset:
            json_dialogs = json.load(dataset)

        if kwargs.get("one_utterance_per_dialog"):
            data = self.one_dialog(json_dialogs)
        else:
            data = self.many_dialogs(json_dialogs)

        return data

    @staticmethod
    def one_dialog(json_dialogs):
        data = {"train": []}
        for json_dialog in json_dialogs:
            history = []
            for i, replica in enumerate(json_dialog["dialog"]):
                if i != len(json_dialog["dialog"]) - 1:
                    history.append(replica['text'])
                else:
                    utterance = replica['text']
            data['train'].append(((utterance, history), None))
        return data

    @staticmethod
    def many_dialogs(json_dialogs):
        data = {"train": []}
        for json_dialog in json_dialogs:
            history = []
            for replica in json_dialog["dialog"]:
                data['train'].append(((replica['text'], history), None))
                history.append(replica['text'])

        return data
