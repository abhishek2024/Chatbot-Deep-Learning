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
    def __init__(self, save_path: str, dim: int, **kwargs) -> None:
        self.dim = dim
        self.save_path = save_path
        self.elmo = ELMoEmbedder(**kwargs)
        if not kwargs.get('concat_last_axis', True):
            self.layer_num = len(kwargs['elmo_output_names'])
            self.pure_layers = True

        super().__init__(save_path=save_path, **kwargs)

    def fit(self, batch: List[List[List[str]]], doc_keys: List[str], *args, **kwargs) -> None:
        with h5py.File(self.save_path, "w") as out_file:
            for doc_num, (doc_sentences, doc_key) in enumerate(zip(batch, doc_keys)):
                text_len = np.array([len(s) for s in doc_sentences])
                tf_lm_emb = self.elmo(doc_sentences)
                group = out_file.create_group(doc_key)

                if self.pure_layers:
                    tf_lm_emb = self._reshape(tf_lm_emb)

                for i, (e, l) in enumerate(zip(tf_lm_emb, text_len)):
                    if self.pure_layers:
                        e = e[:l, :]
                        group[str(i)] = e
                    else:
                        e = e[:l, :, :]
                        group[str(i)] = e

                if doc_num % 10 == 0:
                    print(f"[ Cached {doc_num + 1}/{len(doc_keys)} documents in {self.save_path} ]")

    def _reshape(self, vectors):
        sent_num = len(vectors)
        word_num = max([len(vectors[i]) for i in range(sent_num)])
        emb = np.zeros((sent_num, word_num, self.layer_num, self.dim))
        for i, sent in enumerate(vectors):
            for j, word in enumerate(sent):
                for k, vec in enumerate(word):
                    if len(vec) != self.dim:
                        emb[i, j, k] = np.concatenate([vec, vec], axis=-1)
                    else:
                        emb[i, j, k] = vec

        emb = np.reshape(emb, (sent_num, word_num, self.dim, self.layer_num))
        return emb

    def destroy(self):
        self.elmo.destroy()

    def __call__(self, *args, **kwargs):
        pass

    def reset(self):
        pass

    def save(self, *args, **kwargs):
        pass

    def load(self, *args, **kwargs):
        pass
