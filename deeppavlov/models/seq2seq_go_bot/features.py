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
from typing import List
from logging import getLogger

import numpy as np

from deeppavlov.core.models.estimator import Estimator


log = getLogger()


class StateFeaturizer(Estimator):

    def __init__(self, dontcare_value: str = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dontcare_value = dontcare_value
        self.dim = 2 if dontcare_value else 1
        self.len = 0
        self.keys = []
        if self.load_path.exists():
            self.load()

    @classmethod
    def _get_depth(cls, d: dict) -> int:
        for k in d.keys():
            if isinstance(d[k], dict):
                return cls._get_depth(d[k]) + 1
            break
        return 1

    def _flatten_dict(cls, d: dict, pref="") -> dict:
        if cls._get_depth(d) == 1:
            return {(pref + "_" + k) if pref else k: v
                    for k, v in d.items()}
        new_d = {}
        for k, v in d.items():
            new_k = pref + "_" + k if pref else k
            new_v = cls._flatten_dict(v, pref=new_k)
            key_collision = set(new_v.keys()).intersection(new_d.keys())
            if key_collision:
                raise RuntimeError(f"collision in key names {key_collision}")
            new_d.update(new_v)
        return new_d 

    def _featurize(self, values: List[str]) -> np.ndarray:
        feats = np.zeros((self.len,), dtype=float) 
        for i, key in enumerate(self.keys):
            val = values.get(key, "")
            feats[self.dim * i] = float(val == "")
            if self.dontcare_value:
                feats[self.dim * i + 1] = float(val == self.dontcare_value) 
        return feats

    def fit(self, states: List[dict]) -> None:
        for state in states:
            state_flat = self._flatten_dict(state)
            self.keys = set(state_flat.keys()).union(self.keys)
        self.keys = list(self.keys)
        self.len = len(self.keys) * self.dim
        log.info(f"State features have length = {self.len}.")

    def __call__(self, states: List[dict]) -> List[np.ndarray]:
        feats = []
        for state in states:
            state = state or {}
            state_flat = self._flatten_dict(state)
            f = self._featurize(state_flat)
            feats.append(f)
        return feats

    def save(self, **kwargs):
       json.dump(self.keys, open(self.save_path, 'wt'))

    def load(self, **kwargs):
        self.keys = json.load(open(self.load_path, 'rt'))
        self.len = len(self.keys) * self.dim

