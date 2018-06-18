from typing import List, Tuple, Any
import time

import numpy as np
import tensorflow_hub as hub
import tensorflow as tf

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('sentence_ranker')
class TFHUBSentenceRanker(Component):
    def __init__(self, top_n=20, return_vectors=False, active: bool = True, **kwargs):
        """
        :param top_n: top n sentences to return
        :param return_vectors: return unranged USE vectors instead of sentences
        :param active: when is not active, return all sentences
        """
        self.embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")
        self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                        log_device_placement=True,
                                                        gpu_options=tf.GPUOptions(
                                                            per_process_gpu_memory_fraction=0.1,
                                                            allow_growth=False
                                                        )))
        self.session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        self.q_ph = tf.placeholder(shape=(None,), dtype=tf.string)
        self.c_ph = tf.placeholder(shape=(None,), dtype=tf.string)
        self.q_emb = self.embed(self.q_ph)
        self.c_emb = self.embed(self.c_ph)
        self.top_n = top_n
        self.return_vectors = return_vectors
        self.active = active

    def __call__(self, query_context_id: List[Tuple[str, List[str]]]):
        """
        Rank sentences and return top n sentences.
        """
        predictions = []
        all_top_scores = []
        fake_scores = [0.001] * len(query_context_id)

        for el in query_context_id:
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
                if self.active:
                    thresh = self.top_n
                else:
                    thresh = len(query_context_id)
                if scores.size == 1:
                    top_scores = np.sort([scores])[::-1][:thresh]
                else:
                    top_scores = np.sort(scores)[::-1][:thresh]
                all_top_scores.append(top_scores)
                sentence_top_ids = np.argsort(scores)[::-1][:thresh]
                predictions.append([el[1][x] for x in sentence_top_ids])
        if self.return_vectors:
            return predictions, fake_scores
        else:
            return [' '.join(sentences) for sentences in predictions], all_top_scores
