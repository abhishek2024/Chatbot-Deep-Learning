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

import sys
import pickle as pkl
from logging import getLogger

import numpy as np
import sqlite3

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.serializable import Serializable
from bert_dp.preprocessing import InputFeatures

logger = getLogger(__name__)


@register('response_base_loader')
class ResponseBaseLoader(Serializable):
    """Class for loading a base with text responses (and contexts) and their vector representations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resps = None
        self.resp_features = None
        self.resp_vecs = None
        self.conts = None
        self.cont_features = None
        self.cont_vecs = None
        self.load()

    def load(self):
        if self.load_path is not None:
            resp_file = self.load_path / "ruwiki_par.db"
            if resp_file.exists():
                conn = sqlite3.connect(str(resp_file))
                cur = conn.cursor()
                cur.execute("SELECT * FROM documents")
                raws = cur.fetchall()
                self.resps = [el[2].replace('\n', ' ').replace('#', '') for el in raws][:4]
            else:
                logger.error("Please provide responses.csv file to the {} directory".format(self.load_path))
                sys.exit(1)
            resp_feat_file = self.load_path / "resp_feats.pkl"
            if resp_feat_file.exists():
                with open(resp_feat_file, 'rb') as f:
                    resp_feats = pkl.load(f)
                self.resp_features = self._convert_dicts_to_features(resp_feats)
            resp_vec_file = self.load_path / "resp_vecs.npy"
            if resp_vec_file.exists():
                self.resp_vecs = np.load(resp_vec_file)

    def _convert_dicts_to_features(self, feature_dicts):
        features = []
        for el in feature_dicts[0]:
            unique_id = el['unique_id']
            tokens = el['tokens']
            input_ids = el['input_ids']
            input_mask = [el['input_mask']]
            input_type_ids = [el['input_type_ids']]
            f = InputFeatures(unique_id, tokens, input_ids, input_mask, input_type_ids)
            features.append(f)
        features = [features]
        return features

    def save(self):
        logger.error("The method save of the {} class is not used.".format(self.__class__))
