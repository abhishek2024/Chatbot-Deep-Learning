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

from itertools import chain
from typing import List

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.tf_model import TFModel
from deeppavlov.core.common.log import get_logger

logger = get_logger(__name__)


# @register('ranker_encoder')
class BaseNeuralRankerEncoder(TFModel):
    def __init__(self, **kwargs):
        self.hidden_size_dense_1 = 300
        self.hidden_size_dense_2 = 300
        self.hidden_size_dense_3 = 512
        self.learning_rate = 0.01
        self.question_pad_size = 3
        self.context_pad_size = 30
        # self.emb_size = 512
        self.n_epochs = 0
        # self.batch_size = 2

        self.mode = kwargs.get('mode', None)

        self.sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(
                allow_growth=True
            )))

        self._init_graph()
        self._init_optimizer()
        self.sess.run(tf.global_variables_initializer())

        super().__init__(save_path='/home/olga/ololo')

        # load model if exists:
        # super().__init__(**kwargs)
        # if self.load_path is not None:
        #     self.load()

    def _init_hub_modules(self):
        self.embedder = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")

    def _init_graph(self):
        self._init_placeholders()
        self._init_hub_modules()

        # batch_size = tf.shape(self.c_ph)[0]
        batch_size = 2
        c_embs = tf.reshape(self.c_ph, [-1])
        q_embs = tf.reshape(self.q_ph, [-1])
        c_embs = self.embedder(c_embs)
        q_embs = self.embedder(q_embs)
        # emb_size = tf.shape(c_embs)[1]
        emb_size = 512

        c_embs = tf.stack(c_embs)
        q_embs = tf.stack(q_embs)
        c_embs = tf.reshape(c_embs, shape=[batch_size, self.context_pad_size, emb_size])
        q_embs = tf.reshape(q_embs, shape=[batch_size, self.question_pad_size, emb_size])

        print(f"c_embs shape: {c_embs}")
        print(f"q_embs shape: {q_embs}")

        q_mask = tf.sequence_mask(self.q_len_ph, maxlen=self.question_pad_size, dtype=tf.uint8)
        c_mask = tf.sequence_mask(self.c_len_ph, maxlen=self.context_pad_size, dtype=tf.uint8)
        # q_mask = tf.expand_dims(q_mask, axis=0)
        # c_mask = tf.expand_dims(c_mask, axis=0)

        print(f"q_mask shape: {q_mask}")
        print(f"c mask shape: {c_mask}")

        # q_mask = tf.transpose(q_mask)
        # c_mask = tf.transpose(c_mask)

        q_embs = tf.boolean_mask(q_embs, q_mask, axis=1)
        c_embs = tf.boolean_mask(c_embs, c_mask, axis=1)

        print("Shapes after masking:")
        print(f"c_embs shape: {c_embs}")
        print(f"q_embs shape: {q_embs}")

        c_norm_emb = tf.reduce_sum(c_embs, axis=1, keepdims=True) / tf.sqrt(tf.cast(self.c_len_ph, dtype=tf.float32))
        q_norm_emb = tf.reduce_sum(q_embs, axis=1, keepdims=True) / tf.sqrt(tf.cast(self.q_len_ph, dtype=tf.float32))

        print("Normalized shapes: ")
        print(f"c_norm shape: {c_norm_emb}")
        print(f"q_norm shape: {q_norm_emb}")

        # c_norm_emb = tf.reshape(c_norm_emb, shape=[batch_size, emb_size])

        dense_1 = tf.layers.dense(c_norm_emb, units=self.hidden_size_dense_1, activation=tf.tanh,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
        dense_2 = tf.layers.dense(dense_1, units=self.hidden_size_dense_2, activation=tf.tanh,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
        dense_3 = tf.layers.dense(dense_2, units=self.hidden_size_dense_3, activation=tf.tanh,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
        dot_product = tf.matmul(q_norm_emb, dense_3, transpose_b=True)

        # if self.mode == 'infer':
        self.one_hot_answer = tf.nn.softmax(logits=dot_product)

        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=dot_product, labels=self.y_ph)

        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def _init_placeholders(self):
        self.c_ph = tf.placeholder(shape=(None, None), dtype=tf.string, name='c_ph')
        self.q_ph = tf.placeholder(shape=(None, None), dtype=tf.string, name='q_ph')
        self.c_len_ph = tf.placeholder(shape=(None,), dtype=tf.uint8,
                                       name='c_len_ph')
        self.q_len_ph = tf.placeholder(shape=(None,), dtype=tf.uint8,
                                       name='q_len_ph')
        self.y_ph = tf.placeholder(shape=(None,), dtype=tf.uint8, name='y_ph')

    def _init_optimizer(self):
        with tf.variable_scope('Optimizer'):
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                opt = tf.train.MomentumOptimizer(self.learning_rate, momentum=0)
                grads_and_vars = opt.compute_gradients(self.loss)
                self.train_op = opt.apply_gradients(grads_and_vars)

        # self.train_op = tf.train.MomentumOptimizer(self.learning_rate, 0).minimize(self.loss)

    def _build_feed_dict(self, query, chunk, label=None):
        assert len(query) == len(chunk)
        # if self.batch_size is None:
        #     self.batch_size = len(query)
        q_len = [len(el) for el in query]
        c_len = [len(el) for el in chunk]
        # q_mask = [[1] * len(i) + [0] * (self.question_pad_size - len(i)) for i in query]
        # c_mask = [[1] * len(i) + [0] * (self.context_pad_size - len(i)) for i in chunk]
        # q_pad = self._pad_sentences(query, self.question_pad_size)
        # c_pad = self._pad_sentences(chunk, self.context_pad_size)
        # q_flat = list(chain.from_iterable(q_pad))
        # c_flat = list(chain.from_iterable(c_pad))
        feed_dict = {self.q_ph: query, self.c_ph: chunk, self.q_len_ph: q_len, self.c_len_ph: c_len}
        if label is not None:
            feed_dict[self.y_ph] = label
        return feed_dict

    def train_on_batch(self, query: List[List[str]], chunk: List[List[str]], label: List[int]):
        feed_dict = self._build_feed_dict(query, chunk, label)
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict)
        return loss

    def __call__(self, query: List[List[str]], chunk: List[List[str]]):
        feed_dict = self._build_feed_dict(query, chunk)
        return self.sess.run(self.one_hot_answer, feed_dict)

    @staticmethod
    def _pad_sentences(batch, limit):
        for sentences in batch:
            sentences += [''] * limit
        return batch


q = [["Question 1?"], ["This is question 2?"]]
c = [["Doc 1 sentence 1.", "Doc second sentence second."],
     ["Doc 2 sentence 1.", "Doc second sentence second.", "Third."]]
y = [0, 1]
encoder = BaseNeuralRankerEncoder()
prob = encoder(q, c)
# loss = encoder.train_on_batch(q, c, y)
print(prob)
# print(loss)
