# Copyright 2018 Neural Networks and Deep Learning lab, MIPT
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json

from collections import Counter
from typing import List, Tuple, Dict, Any
from operator import itemgetter
from scipy.stats import entropy
import numpy as np
from scipy.sparse.linalg import norm as sparse_norm
from scipy.sparse import vstack
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm as sparse_norm
from scipy.spatial.distance import cosine, euclidean

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.file import save_pickle, load_pickle
from deeppavlov.core.commands.utils import expand_path, make_all_dirs, is_file_exist
from deeppavlov.core.models.estimator import Component

logger = get_logger(__name__)

@register("ecommerce_tfidf_bot")
class EcommerceTfidfBot(Component):
    """Class to retrieve product items from `load_path` catalogs
    in sorted order according to the similarity measure
    Retrieve the specification attributes with corresponding values
    in sorted order according to entropy.

    Parameters:
        preprocess: text preprocessing component
        save_path: path to save a model
        load_path: path to load a model
        entropy_fields: the specification attributes of the catalog items
        min_similarity: similarity threshold for ranking
        min_entropy: min entropy threshold for specifying
    """

    def __init__(self, 
                save_path: str, 
                load_path: str, 
                entropy_fields: list, 
                min_similarity: float = 0.5,
                min_entropy: float = 0.5, 
                **kwargs) -> None:

        self.save_path = expand_path(save_path)
        self.load_path = expand_path(load_path)

        self.min_similarity = min_similarity
        self.min_entropy = min_entropy
        self.entropy_fields = entropy_fields
        self.ec_data: List = []
        if kwargs.get('mode') != 'train':
            self.load()

    def fit(self, data, query) -> None:

        # print(query)
        # pass
        """Preprocess items `title` and `description` from the `data`

        Parameters:
            data: list of catalog items

        Returns:
            None
        """
    
        self.x_train_features = vstack(list(query))
        self.ec_data = data
        
    def save(self) -> None:
        """Save classifier parameters"""
        logger.info("Saving to {}".format(self.save_path))
        path = expand_path(self.save_path)
        make_all_dirs(path)

#        print("densing")
 #       self.x_train_features = [x.todense() for x in self.x_train_features]

        save_pickle((self.ec_data, self.x_train_features), path)

    def load(self) -> None:
        """Load classifier parameters"""
        logger.info("Loading from {}".format(self.load_path))
        self.ec_data, self.x_train_features = load_pickle(expand_path(self.load_path))

 #       print("densing")
  #      self.x_train_features = [x.todense() for x in self.x_train_features]

    def __call__(self, q_vects, histories, states):

        logger.info(f"Total catalog {len(self.ec_data)}")

        print(q_vects)

        if not isinstance(q_vects, list):
            print("converted into list")
            q_vects = [q_vects]

        if not isinstance(states, list):
            states = [states]

        if not isinstance(histories, list):
            histories = [histories]

        items: List = []
        confidences: List = []
        states: List = []

        for idx, q_vect in enumerate(q_vects):

            if len(states)>=idx+1:
                state = states[idx]
            else:
                state = {'start': 0, 'stop': 5}
            
            if 'start' not in state:
                state['start'] = 0
            if 'stop' not in state:
                state['stop'] = 5

#            q_vect_dense = q_vect.todense()

            #cos_distance = [cosine(q_vect_dense, x.todense()) for x in self.x_train_features]
            print("cosining")
            #cos_distance = [cosine(q_vect_dense, x) for x in self.x_train_features]
            norm = sparse_norm(q_vect) * sparse_norm(self.x_train_features, axis=1)
            cos_similarities = np.array(q_vect.dot(self.x_train_features.T).todense())/norm

            cos_similarities = cos_similarities[0]
            cos_similarities = np.nan_to_num(cos_similarities)
        
   #         scores = [(cos, len(self.ec_data[idx]['Title'])) for idx, cos in enumerate(cos_distance)]
    #        print("calc cosine")

    #        raw_scores = np.array(scores, dtype=[('x', 'float_'), ('y', 'int_')])

     #       answer_ids = np.argsort(raw_scores, order=('x', 'y'))
      #      print("sorted")
            answer_ids = np.argsort(cos_similarities)[::-1]

            # results_args_sim = [idx for idx in results_args if scores[idx] >= self.min_similarity]
        
            items.append([self.ec_data[idx] for idx in answer_ids[state['start']:state['stop']]])

            #confidences.append([cos_distance[idx] for idx in answer_ids[state['start']:state['stop']]])
            confidences.append([cos_similarities[idx] for idx in answer_ids[state['start']:state['stop']]])

            states.append(state)

        print(items)
        print(confidences)

        return items, confidences, states
