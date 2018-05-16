from typing import List, Tuple
import time

import numpy as np
import tensorflow_hub as hub
import tensorflow as tf

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('sentence_ranker')
class TFHUBSentenceRanker(Component):
    def __init__(self, top_k=20, **kwargs):
        self.embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")
        self.session = tf.Session()
        self.session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        self.q_ph = tf.placeholder(shape=(None,), dtype=tf.string)
        self.c_ph = tf.placeholder(shape=(None,), dtype=tf.string)
        self.q_emb = self.embed(self.q_ph)
        self.c_emb = self.embed(self.c_ph)
        self.top_k = top_k

    def __call__(self, query_cont: List[Tuple[str, List[str]]]):
        predictions = []
        for el in query_cont:
            # DEBUG
            # start_time = time.time()
            qe, ce = self.session.run([self.q_emb, self.c_emb],
                                      feed_dict={
                                          self.q_ph: [el[0]],
                                          self.c_ph: el[1],
                                      })
            # print("Time spent: {}".format(time.time() - start_time))
            scores = (qe @ ce.T).squeeze()
            top_ids = np.argsort(scores)[::-1][:self.top_k]
            predictions.append([el[1][x] for x in top_ids])
        res = [' '.join(sentences) for sentences in predictions]
        return res

