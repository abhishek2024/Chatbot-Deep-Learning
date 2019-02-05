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

from typing import List, Union

import h5py
import numpy as np

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

log = get_logger(__name__)


@register("elmo_cache_embedder")
class ElmoCacheEmbedder(Component):
    def __init__(self, cache_file: str, dim: int, mean: bool = False, **kwargs) -> None:
        self.cache_file = str(expand_path(cache_file))
        self.lm_file = h5py.File(self.cache_file, "r")
        self.dim = dim
        self.mean = mean

    def __call__(self, doc_key, *args, **kwargs) -> Union[List[np.ndarray], np.ndarray]:
        group = self.lm_file[doc_key]
        num_sentences = len(list(group.keys()))
        sentences = [group[str(i)][...] for i in range(num_sentences)]

        if not self.mean:
            lm_emb = np.zeros([num_sentences, max(s.shape[0] for s in sentences), self.dim])
            for i, s in enumerate(sentences):
                lm_emb[i, :s.shape[0], :] = s
        else:
            lm_emb = np.zeros([num_sentences, self.dim])
            for i, s in enumerate(sentences):
                filtered = [et for et in s if np.any(et)]
                if filtered:
                    return np.mean(filtered, axis=0)
                lm_emb[i] = s
        return lm_emb

    def reset(self):
        pass

    def destroy(self):
        del self.lm_file
