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

from logging import getLogger
from typing import Any, Dict, List, Tuple

from sklearn.model_selection import train_test_split

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator

log = getLogger(__name__)


@register("coref_dialog_iterator")
class CorefDialogIterator(DataLearningIterator):
    def __init__(self, data: Dict[str, List[Tuple[Any, Any]]], seed: int = None, shuffle: bool = True,
                 *args, **kwargs) -> None:
        super().__init__(data, seed, shuffle, *args, **kwargs)

    def split(self, split_fields: List[str] = None, split_proportions: List[float] = None,
              split_seed: int = None, stratify: bool = None, mode: str = 'infer') -> None:
        if split_fields is not None:
            if "valid" not in split_fields and "test" not in split_fields or len(split_fields) > 3:
                raise ValueError("Given split fields not contain 'valid' or 'test' names, "
                                 "or len of split fields more than 3.")

            log.info("Splitting field <<{}>> to new fields <<{}>>".format("train", split_fields))

            if split_seed is None:
                split_seed = self.random.randint(0, 10000)
            data_to_div = self.train.copy()
            data_size = len(self.train)

            new_data = {}
            for i in range(len(split_fields) - 1):
                if stratify:
                    stratify = [sample[1] for sample in data_to_div]

                new_data[split_fields[i]], data_to_div = train_test_split(
                    data_to_div,
                    test_size=len(data_to_div) - int(data_size * split_proportions[i]),
                    random_state=split_seed,
                    stratify=stratify)
                new_data[split_fields[-1]] = data_to_div

            self.train = new_data.get("train", [])
            self.valid = new_data.get("valid", [])
            self.test = new_data.get("test", [])
