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
from typing import Dict, List

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.models.coreference_resolution.old_model.conll2model_format import conll2modeldata
from deeppavlov.models.coreference_resolution.old_model.rucor2conll import rucoref2conll, split_doc


@register("coreference_reader")
class CorefReader(DatasetReader):
    def __init__(self):
        self.data_path = None
        self.folder_path = None
        self.dataset_name = None

    def read(self, data_path: str, *args, **kwargs) -> Dict[str, List[str]]:
        self.data_path = Path(data_path)
        self.folder_path = self.data_path.parent
        self.dataset_name = "rucoref_conll"

        if not self.data_path.exists():
            raise ValueError(f"Russian Coreference dataset in a {self.data_path} folder is absent.")

        return self.read_jsonlines()

    def rucor2conll(self) -> None:
        rucoref2conll(self.data_path, self.folder_path)
        self.folder_path.joinpath(self.dataset_name).mkdir(exist_ok=False)
        split_doc(self.folder_path.joinpath('russian.v4_conll'),
                  self.folder_path.joinpath(self.dataset_name))
        self.folder_path.joinpath('russian.v4_conll').unlink()

    def read_jsonlines(self) -> Dict:
        train_path = self.folder_path.joinpath("train.jsonl")
        if not train_path.exists():
            if not self.folder_path.joinpath(self.dataset_name).exists():
                if not self.folder_path.joinpath("rucoref_conll").exists():
                    self.rucor2conll()

            for path in sorted(self.folder_path.joinpath(self.dataset_name).glob("*.v4_conll")):
                with path.open("r", encoding='utf8') as conll_file:
                    conll_str = conll_file.read()

                with train_path.open('a', encoding='utf8') as train_file:
                    print(json.dumps(conll2modeldata(conll_str)), file=train_file)

        return dict(train=[train_path])
