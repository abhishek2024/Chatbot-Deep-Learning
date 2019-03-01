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

from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.errors import ConfigError

log = getLogger(__name__)


@register('char_splitter')
class CharSplitter(Component):
    """This component transforms batch of sequences of tokens into batch of sequences of character sequences or \
        batch of strings into batch of sequences of characters"""
    def __init__(self, **kwargs) -> None:
        pass

    @overrides
    def __call__(self, batch, *args, **kwargs):
        char_batch = []
        if isinstance(batch[0], list):
            for tokens_sequence in batch:
                char_batch.append([list(tok) for tok in tokens_sequence])
            return char_batch
        elif isinstance(batch[0], str):
            for sentence in batch:
                char_batch.append(list(sentence))
            return char_batch
        else:
            raise ConfigError("CharSplitter can not split given data type: {}".format(type(batch[0])))
