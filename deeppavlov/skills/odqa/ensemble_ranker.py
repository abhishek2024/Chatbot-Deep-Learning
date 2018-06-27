from typing import List, Union, Any
from operator import itemgetter
from itertools import chain

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.component import Component
from .utils import flatten_nested_list

logger = get_logger(__name__)


@register("ensemble_ranker")
class EnsembleRanker(Component):

    def __init__(self, top_n=10, active=True, *args, **kwargs):
        self.top_n = top_n
        self.active = active

    def __call__(self, tfidf: List[List[List[Union[Any]]]],
                 tfhub: List[List[List[Union[Any]]]],
                 rnet: List[List[List[Union[Any]]]], *args, **kwargs) -> \
            List[List[List[Union[str, int, float]]]]:

        CHUNK_ID = 3
        SCORE_IDX = 2
        NUM_RANKERS = 3
        FAKE_SCORE = 0.001

        def update_all_predictions(predictions, ranker_instance):
            for predicted_chunk in ranker_instance:
                chunk_idx = predicted_chunk[CHUNK_ID]
                if chunk_idx in instance_data_ids:
                    data_idx = list(map(itemgetter(CHUNK_ID), predictions)).index(chunk_idx)
                    predictions[data_idx][SCORE_IDX] = flatten_nested_list(
                        predictions[data_idx][SCORE_IDX] + [predicted_chunk[SCORE_IDX]])
                else:
                    predicted_chunk[SCORE_IDX] = [predicted_chunk[SCORE_IDX]]
                    predictions.append(predicted_chunk)

        # Normalize tfidf scores
        for instance in tfidf:
            scores = list(map(itemgetter(SCORE_IDX), instance))
            norm = np.linalg.norm(scores)
            for prediction in instance:
                prediction[SCORE_IDX] = float(prediction[SCORE_IDX] / norm)

        # Count average scores from all rankers
        all_data = []
        for tfidf_instance, tfhub_instance, rnet_instance in zip(tfidf, tfhub, rnet):

            for item in tfidf_instance:
                item[SCORE_IDX] = [item[SCORE_IDX]]

            instance_predictions = tfidf_instance
            instance_data_ids = set(map(itemgetter(CHUNK_ID), tfidf_instance))

            update_all_predictions(instance_predictions, tfhub_instance)
            update_all_predictions(instance_predictions, rnet_instance)

            for prediction in instance_predictions:
                len_scores = len(prediction[SCORE_IDX])
                assert len_scores <= NUM_RANKERS
                if len_scores < NUM_RANKERS:
                    prediction[SCORE_IDX] = np.mean(
                        prediction[SCORE_IDX] + (NUM_RANKERS - len_scores) * [FAKE_SCORE])
                else:
                    prediction[SCORE_IDX] = np.mean(prediction[SCORE_IDX])

            instance_predictions = sorted(instance_predictions, key=itemgetter(SCORE_IDX), reverse=True)

            if self.active:
                instance_predictions = instance_predictions[:self.top_n]

            for i in range(len(instance_predictions)):
                instance_predictions[i][0] = i

            all_data.append(instance_predictions)

        return all_data
