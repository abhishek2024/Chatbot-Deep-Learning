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

from typing import List, Callable
from collections import Counter, defaultdict
import itertools
from pathlib import Path

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.component import Component

log = get_logger(__name__)


@register('char_connector')
class CharConnector(Component):
    def __init__(self,
                 save_path: str = None,
                 load_path: str = None,
                 **kwargs) -> None:
        pass

    def __call__(self, batch: List[List[str]], **kwargs) -> List[str]:
        connected_batch = []
        for sample in batch:
            sentence = ""
            for char in batch:
                sentence += char
            connected_batch.append(sentence)
        return connected_batch
