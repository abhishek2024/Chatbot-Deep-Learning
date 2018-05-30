from typing import List, Tuple
import time

import numpy as np
import tensorflow_hub as hub
import tensorflow as tf

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('sentence_ranker')
class TFHUBSentenceRanker(Component):
    def __init__(self, top_n=20, return_vectors=False, active: bool=True, **kwargs):
        """
        :param top_n: top n sentences to return
        :param return_vectors: return unranged USE vectors instead of sentences
        :param active: when is not active, return all sentences
        """
        self.embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")
        self.session = tf.Session()
        self.session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        self.q_ph = tf.placeholder(shape=(None,), dtype=tf.string)
        self.c_ph = tf.placeholder(shape=(None,), dtype=tf.string)
        self.q_emb = self.embed(self.q_ph)
        self.c_emb = self.embed(self.c_ph)
        self.top_k = top_n
        self.return_vectors = return_vectors
        self.active = active

    def __call__(self, query_cont: List[Tuple[str, List[str]]]):
        """
        Rank sentences and return top n sentences.
        """
        predictions = []

        if self.active:
            for el in query_cont:
                # DEBUG
                # start_time = time.time()
                qe, ce = self.session.run([self.q_emb, self.c_emb],
                                          feed_dict={
                                              self.q_ph: [el[0]],
                                              self.c_ph: el[1],
                                          })
                # print("Time spent: {}".format(time.time() - start_time))
                if self.return_vectors:
                    predictions.append((qe, ce))
                else:
                    scores = (qe @ ce.T).squeeze()
                    top_ids = np.argsort(scores)[::-1][:self.top_k]
                    predictions.append([el[1][x] for x in top_ids])
            if self.return_vectors:
                return predictions
            else:
                return [' '.join(sentences) for sentences in predictions]
        else:
            docs = [' '.join(item[1]) for item in query_cont]
            return docs

