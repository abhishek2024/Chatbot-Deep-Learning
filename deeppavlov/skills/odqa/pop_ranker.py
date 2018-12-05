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

from typing import List, Union, Tuple, Any
from operator import itemgetter
from itertools import chain
import copy

from sklearn.externals import joblib
import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.estimator import Component
from deeppavlov.core.common.file import read_json

logger = get_logger(__name__)


@register("pop_ranker")
class PopRanker(Component):
    """Get documents as input and output smaller number of the best documents according to their popularity.
    """

    def __init__(self, pop_dict_path: str, load_path: str, top_n: int = 3, active: bool = True, **kwargs):
        logger.info(f"Reading popularity dictionary from {pop_dict_path}")
        self.pop_dict = read_json(pop_dict_path)
        logger.info(f"Loading popularity ranker from {load_path}")
        self.clf = joblib.load(load_path)
        self.top_n = top_n
        self.active = active

    def __call__(self, rank_text_score_id):
        """
        Get tfidf scores and article titles, output smaller number of best articles.
        """
        TFIDF_SCORE_IDX = 2
        TITLE_IDX = 3
        CLF_PROBA_IDX = 4
        reranked = copy.deepcopy(rank_text_score_id)
        for ra_list in reranked:
            for j in range(len(ra_list)):
                pop = self.pop_dict[ra_list[j][TITLE_IDX]]
                tfidf_score = ra_list[j][TFIDF_SCORE_IDX]
                features = [tfidf_score, pop, tfidf_score * pop]
                prob = self.clf.predict_proba([features])
                ra_list[j] = tuple((*ra_list[j], prob[0][1]))

        for i in range(len(reranked)):
            reranked[i] = sorted(reranked[i], key=itemgetter(CLF_PROBA_IDX, TFIDF_SCORE_IDX), reverse=True)
            if self.active:
                reranked[i] = reranked[i][:self.top_n]

        return reranked


@register("pop_title_ranker")
class PopTitleRanker(Component):
    """Get documents as input and output smaller number of the best documents according to their popularity.
    """

    def __init__(self, pop_dict_path: str, load_path: str, top_n: int = 3, active: bool = True, **kwargs):
        logger.info(f"Reading popularity dictionary from {pop_dict_path}")
        self.pop_dict = read_json(pop_dict_path)
        logger.info(f"Loading popularity ranker from {load_path}")
        self.clf = joblib.load(load_path)
        self.top_n = top_n
        self.active = active

    def __call__(self, rank_text_score_id_title):
        """
        Get tfidf scores and article titles, output smaller number of best articles.
        """
        TFIDF_SCORE_IDX = 2
        TITLE_IDX = 3
        TITLE_SCORE_IDX = 4
        CLF_PROBA_IDX = 5
        reranked = copy.deepcopy(rank_text_score_id_title)
        for ra_list in reranked:
            for j in range(len(ra_list)):
                title = ra_list[j][TITLE_IDX]
                title_len = len(title.split())
                pop = self.pop_dict[title]
                tfidf_score = ra_list[j][TFIDF_SCORE_IDX]
                title_score = ra_list[j][TITLE_SCORE_IDX]
                # features = [tfidf_score, pop, tfidf_score * pop, tfidf_score * title_score, title_score * pop]
                # features = [tfidf_score, pop, title_score, tfidf_score * pop, title_score * pop, title_len]
                # features = [tfidf_score, pop, tfidf_score * title_score * pop / 10000]
                features = [tfidf_score, pop, tfidf_score * pop,  title_len]

                prob = self.clf.predict_proba([features])
                ra_list[j] = tuple((*ra_list[j], prob[0][1]))

        for i in range(len(reranked)):
            reranked[i] = sorted(reranked[i], key=itemgetter(CLF_PROBA_IDX, TFIDF_SCORE_IDX), reverse=True)
            if self.active:
                reranked[i] = reranked[i][:self.top_n]

        return reranked
