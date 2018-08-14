from typing import List, Tuple, Any
import time

import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from scipy.spatial.distance import cosine

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('elmo_ranker')
class ELMoRanker(Component):
    def __init__(self, top_n=20, return_vectors=False, active: bool = True, **kwargs):

        self.embed = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
        self.session = tf.Session()
        self.session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        self.q_ph = tf.placeholder(shape=(None,), dtype=tf.string)
        self.c_ph = tf.placeholder(shape=(None,), dtype=tf.string)
        self.q_emb = self.embed(self.q_ph, signature='default', as_dict=True)['default']
        self.c_emb = self.embed(self.c_ph, signature='default', as_dict=True)['default']
        self.top_n = top_n
        self.return_vectors = return_vectors
        self.active = active

    def __call__(self, query_context: List[Tuple[str, str]]):

        predictions = []
        all_top_scores = []
        fake_scores = [0.001] * len(query_context)

        for query, context in query_context:
            # DEBUG
            # start_time = time.time()

            qe, ce = self.session.run([self.q_emb, self.c_emb], feed_dict={self.q_ph: [query], self.c_ph: [context]})
            # print("Time spent: {}".format(time.time() - start_time))
            if self.return_vectors:
                predictions.append((qe, ce))
            else:
                score = cosine(qe, ce)
                all_top_scores.append(score)
                predictions.append(context)
        if self.return_vectors:
            return predictions, fake_scores
        else:
            return predictions, all_top_scores
