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

from typing import List

import h5py
import numpy as np

from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Estimator
from deeppavlov.models.embedders.elmo_embedder import ELMoEmbedder

log = get_logger(__name__)


@register('elmo_cahed_estimator')
class ELMoEstimator(Estimator):
    def __init__(self, save_path: str, **kwargs) -> None:
        self.save_path = save_path
        self.elmo = ELMoEmbedder(**kwargs)
        super().__init__(save_path=save_path, **kwargs)

    def fit(self, batch: List[List[List[str]]], doc_keys: List[str], *args, **kwargs) -> None:
        with h5py.File(self.save_path, "w") as out_file:
            for doc_num, (doc_sentences, doc_key) in enumerate(zip(batch, doc_keys)):
                text_len = np.array([len(s) for s in doc_sentences])

                tf_lm_emb = self.elmo(doc_sentences)  # Union[List[np.ndarray], np.ndarray]:

                group = out_file.create_group(doc_key)

                for i, (e, l) in enumerate(zip(tf_lm_emb, text_len)):
                    e = e[:l, :]
                    group[str(i)] = e

                if doc_num % 10 == 0:
                    print(f"[ Cached {doc_num + 1}/{len(doc_keys)} documents in {self.save_path} ]")

    def __call__(self, *args, **kwargs):
        pass

    def reset(self):
        pass

    def destroy(self):
        self.elmo.destroy()

    def save(self, *args, **kwargs):
        pass

    def load(self, *args, **kwargs):
        pass
