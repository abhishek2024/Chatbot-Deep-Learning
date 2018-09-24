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

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.file import save_pickle, load_pickle
from deeppavlov.core.commands.utils import expand_path, make_all_dirs, is_file_exist
from deeppavlov.core.models.estimator import Component
from deeppavlov.metrics.bleu import bleu_advanced

log = get_logger(__name__)

@register("ecommerce_bot")
class EcommerceBot(Component):
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

    def __init__(self, preprocess: Component, save_path: str, load_path: str, 
                 entropy_fields: list, min_similarity: float = 0.5,
                 min_entropy: float = 0.5, **kwargs) -> None:
        self.preprocess = preprocess
        self.save_path = expand_path(save_path)

        if isinstance(load_path, list):
            self.load_path = [expand_path(path) for path in load_path]
        else:
            self.load_path = [expand_path(load_path)]

        self.min_similarity = min_similarity
        self.min_entropy = min_entropy
        self.entropy_fields = entropy_fields
        self.ec_data: List = []
        if kwargs.get('mode') != 'train':
            self.load()

    def fit(self, data: List[Dict[Any, Any]]) -> None:
        """Preprocess items `title` and `description` from the `data`

        Parameters:
            data: list of catalog items

        Returns:
            None
        """

        log.info(f"Items to nlp: {len(data)}")
        self.ec_data = [dict(item, **{
                                    'title_nlped': self.preprocess.spacy2dict(self.preprocess.analyze(item['Title'])),
                                    # 'feat_nlped': self.preprocess.spacy2dict(self.preprocess.analyze(item['Feature']))
                                    'feat_nlped': self.preprocess.spacy2dict(self.preprocess.analyze(item['Title']+'. '+item['Feature']))
                                      }) for item in data]
        log.info('Data are nlped')

    def save(self, **kwargs) -> None:
        """Save classifier parameters"""
        log.info(f"Saving model to {self.save_path}")
        make_all_dirs(self.save_path)
        save_pickle(self.ec_data, self.save_path)

    def load(self, **kwargs) -> None:
        """Load classifier parameters"""
        log.info(f"Loading model from {self.load_path}")
        for path in self.load_path:
            if is_file_exist(path):
                self.ec_data += load_pickle(path)
            else:
                raise Exception(f"File {path} does not exist")
                # log.info(f"File {path} does not exist")

        log.info(f"Loaded items {len(self.ec_data)}")

    def __call__(self, queries: List[str], states: List[Dict[Any, Any]], **kwargs) -> Tuple[Tuple[List[Dict[Any, Any]], List[Any], int], List[float], Dict[Any, Any]]:
        """Retrieve catalog items according to the BLEU measure

        Parameters:
            queries: list of queries
            states: list of dialog state

        Returns:
            response:   items:      list of retrieved items 
                        total:      total number of relevant items
                        entropies:  list of entropy attributes with corresponding values

            confidence: list of similarity scores
            state: dialog state
        """

        response: List = []
        confidence: List = []
        results_args: List = []
        results_args_sim: List = []

        log.debug(f"queries: {queries} states: {states}")

        for item_idx, query_or in enumerate(queries):
           
            state = states[item_idx]
            log.debug(f"type {type(state)}")

            if not isinstance(state, dict):
                log.debug(f"error type {type(state)}")
                state = json.loads(states[item_idx])

            if isinstance(state, str):
                try:
                    state = json.loads(state)
                except:
                    state = self.preprocess.parse_input(state)

            query = self.preprocess.analyze(query_or)

            if 'history' not in state:
                state['history'] = []

            if len(state['history'])>0:

                if query[0].tag_ == 'IN':
                    query = self.preprocess.analyze(state['history'][-1] + " " + query_or)

                elif state['history'][-1] != query_or:
                    cur_prev = self.preprocess.analyze(state['history'][-1] + " " + query_or)
                    complex_bool = self._take_complex_query(cur_prev, query)

                    if complex_bool is True:
                        query = cur_prev
                    else:
                    # current short query wins that means that the state should be zeroed
                        state = {
                        'history': [],
                        'start': 0,
                        'stop': 5,
                        }

            state['history'].append(query.text)

            # start processing the query
            log.debug(f"query: {query}")
            log.debug(f"state: {state}")

            start = state['start'] if 'start' in state else 0
            stop = state['stop'] if 'stop' in state else 5

            query, money_range = self.preprocess.extract_money(query)
            log.debug(f"money detected: {query} {money_range}")

            if len(money_range) == 2:
                state['Price'] = money_range

            # score_title = [bleu_advanced(self.preprocess.lemmas(item['title_nlped']),
            #                self.preprocess.lemmas(self.preprocess.filter_nlp_title(query)),
            #                weights=(1,), penalty=False) for item in self.ec_data]

            # score_feat = [bleu_advanced(self.preprocess.lemmas(item['feat_nlped']),
            #               self.preprocess.lemmas(self.preprocess.filter_nlp(query)),
            #               weights=(0.3, 0.7), penalty=True) for idx, item in enumerate(self.ec_data)]

            # scores = np.mean(
            #     [score_feat, score_title], axis=0).tolist()

            scores = self._similarity(query)

            scores_agg = [np.sum(score[0]) for score in scores]

            scores_title = [(score, -len(self.ec_data[idx]['Title']))
                            for idx, score in enumerate(scores_agg)]

            raw_scores_ar = np.array(scores_title, dtype=[
                ('x', 'float_'), ('y', 'int_')])

            results_args = np.argsort(raw_scores_ar, order=('x', 'y'))[
                ::-1].tolist()

            results_args_sim = [
                idx for idx in results_args if scores_agg[idx] >= self.min_similarity]

            log.debug(f"Items before similarity filtering {len(results_args)} and after {len(results_args_sim)} with th={self.min_similarity} "+
                      f"the best one has score {scores_agg[results_args[0]]} with title {self.ec_data[results_args[0]]['Title']}")

            for key, value in state.items():
                log.debug(f"Filtering for {key}:{value}")

                if key == 'Price':
                    price = value
                    log.debug(f"Items before price filtering {len(results_args_sim)} with price {price}")
                    results_args_sim = [idx for idx in results_args_sim
                                        if self.preprocess.price(self.ec_data[idx]) >= price[0] and
                                        self.preprocess.price(self.ec_data[idx]) <= price[1] and
                                        self.preprocess.price(self.ec_data[idx]) != 0]
                    log.debug(f"Items after price filtering {len(results_args_sim)}")

                elif key in ['query', 'start', 'stop', 'history']:
                    continue

                else:
                    results_args_sim = [idx for idx in results_args_sim
                                        if key in self.ec_data[idx]
                                        if self.ec_data[idx][key].lower() == value.lower()]

            response = []
            for idx in results_args_sim[start:stop]:
                temp = copy.copy(self.ec_data[idx])
                del temp['title_nlped']
                del temp['feat_nlped']
                response.append(temp)

            confidence = [scores[idx][0] for idx in results_args_sim[start:stop]]

            entropies = self._entropy_subquery(results_args_sim)

            state['start'] = start
            state['stop'] = start+len(response)
            if 'history' not in state:
                state['history'] = []

            log.debug(f"Total number of relevant answers {len(results_args_sim)}")

            print(confidence)

        return (response, entropies, len(results_args_sim)), confidence, state

    def _take_complex_query(self, previous, current) -> str:
        coef = 0.05      

        log.debug(f"num of long query {previous} {len(previous)}")
        log.debug(f"num of short query {current} {len(current)}")

        # prev_sim = [score*len(previous)*coef for score in self._similarity(previous)[0]]
        prev_sim = self._similarity(previous)
        prev_sim_agg = [np.sum(score[0]) for score in prev_sim]
        # cur_sim = [score*len(current)*coef for score in self._similarity(current)[0]]
        
        cur_sim = self._similarity(current)
        cur_sim_agg = [np.sum(score[0]) for score in cur_sim]

        prev_arg_max = np.argmax(prev_sim_agg)
        cur_arg_max = np.argmax(cur_sim_agg)

        # prev_sim_sum = np.sum(sorted([score for score in prev_sim], reverse=True)[0])
        # cur_sim_sum = np.sum(sorted([score for score in cur_sim], reverse=True)[0])
        # cur_sim_sum = np.sum([score for score in cur_sim])

        log.debug(f"num of long query {prev_sim[prev_arg_max]}")
        log.debug(f"num of short query {cur_sim[cur_arg_max]}")

        if prev_sim_agg[prev_arg_max]>cur_sim_agg[cur_arg_max]:
            return True

        return False

    def _similarity(self, query: List[Any]) -> List[float]:
        score_title = [bleu_advanced(self.preprocess.lemmas(item['title_nlped']),
                           self.preprocess.lemmas(self.preprocess.filter_nlp_title(query)),
                           weights=(1,), penalty=False, bp_weight=10) for item in self.ec_data]

        score_feat = [bleu_advanced(self.preprocess.lemmas(item['feat_nlped']),
                          self.preprocess.lemmas(self.preprocess.filter_nlp(query)),
                          weights=(0.3, 0.7), penalty=True, bp_weight=10) for idx, item in enumerate(self.ec_data)]
        
        scores = [((score_title[idx], score_feat[idx]), item['Title'], item['Feature']) for idx, item in enumerate(self.ec_data)]
        # confidence = list(zip(score_title, score_feat))

        # scores = np.mean(
                # [score_feat, score_title], axis=0).tolist()

        return scores

    def _entropy_subquery(self, results_args: List[int]) -> List[Tuple[float, str, List[Tuple[str, int]]]]:
        """Calculate entropy of selected attributes for items from the catalog.

        Parameters:
            results_args: items id to consider

        Returns:
            entropies: entropy score with attribute name and corresponding values
        """

        ent_fields: Dict = {}

        for idx in results_args:
            for field in self.entropy_fields:
                if field in self.ec_data[idx]:
                    if field not in ent_fields:
                        ent_fields[field] = []

                    ent_fields[field].append(self.ec_data[idx][field].lower())

        entropies = []
        for key, value in ent_fields.items():
            count = Counter(value)
            entropies.append(
                (entropy(list(count.values()), base=2), key, count.most_common()))

        entropies = sorted(entropies, key=itemgetter(0), reverse=True)
        entropies = [ent_item for ent_item in entropies if ent_item[0]
                     >= self.min_entropy]

        return entropies
