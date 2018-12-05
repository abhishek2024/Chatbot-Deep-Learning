from typing import List, Union, Any
from operator import itemgetter
import copy
import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.component import Component
from .utils import flatten_nested_list

logger = get_logger(__name__)


@register("ensemble_ranker")
class EnsembleRanker(Component):

    def __init__(self, ranker_keys, top_n=10, active=True, *args, **kwargs):
        self.top_n = top_n
        self.active = active
        self.ranker_keys = ranker_keys

    def __call__(self, *args) -> List[List[List[Union[str, int, float]]]]:

        if len(args) != len(self.ranker_keys):
            raise RuntimeError(f'{self.__class__.__name__} ranker_keys and input args length don\'t match:'
                               f' {len(self.ranker_keys)} and {len(args)}')

        CHUNK_IDX = 3
        SCORE_IDX = 2
        FAKE_SCORE = 0.001
        NORM_THRESH = 50  # take only the first 50 results to count np.linalg.norm?

        def update_all_predictions(predictions, ranker_instance):
            for predicted_chunk in ranker_instance:
                chunk_idx = predicted_chunk[CHUNK_IDX]
                if chunk_idx in instance_data_ids:
                    data_idx = list(map(itemgetter(CHUNK_IDX), predictions)).index(chunk_idx)
                    predictions[data_idx][SCORE_IDX] = flatten_nested_list(
                        predictions[data_idx][SCORE_IDX] + [predicted_chunk[SCORE_IDX]])
                else:
                    predicted_chunk[SCORE_IDX] = [predicted_chunk[SCORE_IDX]]
                    predictions.append(predicted_chunk)

        def normalize_scores(ranker_results):
            """
            Normalize paragraph scores with np.linalg.norm
            """
            for instance in ranker_results:
                scores = list(map(itemgetter(SCORE_IDX), instance))
                norm = np.linalg.norm(scores[:NORM_THRESH])
                for pred in instance:
                    pred[SCORE_IDX] = float(pred[SCORE_IDX] / norm)

        tfidf = None
        tfhub = None
        rnet = None

        for ranker_key, arg in zip(self.ranker_keys, args):
            if ranker_key == 'tfidf':
                tfidf = [[list(el) for el in instance] for instance in arg]
                normalize_scores(tfidf)
            elif ranker_key == 'tfhub':
                tfhub = [[list(el) for el in instance] for instance in arg]
            elif ranker_key == 'rnet':
                rnet = [[list(el) for el in instance] for instance in arg]
                normalize_scores(rnet)

        rankers = [r for r in [tfidf, tfhub, rnet] if r is not None]
        num_rankers = len(rankers)

        # Count average scores from all rankers
        all_data = []
        for instances in zip(*rankers):

            for item in instances[0]:
                item[SCORE_IDX] = [item[SCORE_IDX]]

            instance_predictions = copy.deepcopy(instances[0])
            instance_data_ids = set(map(itemgetter(CHUNK_IDX), instance_predictions))  # only unique ids are available!

            for i in range(1, len(instances)):
                update_all_predictions(instance_predictions, instances[i])

            for prediction in instance_predictions:
                len_scores = len(prediction[SCORE_IDX])
                # assert len_scores <= num_rankers
                if len_scores < num_rankers:
                    prediction[SCORE_IDX] = np.mean(
                        prediction[SCORE_IDX] + (num_rankers - len_scores) * [FAKE_SCORE])
                else:
                    prediction[SCORE_IDX] = np.mean(prediction[SCORE_IDX])

            instance_predictions = sorted(instance_predictions, key=itemgetter(SCORE_IDX), reverse=True)

            if self.active:
                instance_predictions = instance_predictions[:self.top_n]

            for i in range(len(instance_predictions)):
                instance_predictions[i][0] = i

            all_data.append(instance_predictions)

        return all_data
