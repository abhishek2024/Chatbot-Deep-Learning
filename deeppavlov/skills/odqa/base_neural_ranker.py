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
import copy
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
        self.emb_size = 512
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

        batch_size = tf.shape(self.c_ph)[0]
        # batch_size = 2
        c_embs = tf.reshape(self.c_ph, [-1])
        # q_embs = tf.reshape(self.q_ph, [-1])
        c_embs = self.embedder(c_embs)
        q_embs = self.embedder(self.q_ph)
        # emb_size = tf.shape(c_embs)[1]

        c_embs = tf.stack(c_embs)
        q_embs = tf.stack(q_embs)
        c_embs = tf.reshape(c_embs, shape=[batch_size, self.context_pad_size, self.emb_size])
        # q_embs = tf.reshape(q_embs, shape=[batch_size, self.question_pad_size, self.emb_size])

        print(f"c_embs shape: {c_embs}")
        print(f"q_embs shape: {q_embs}")

        q_mask = tf.sequence_mask(self.q_len_ph, maxlen=self.question_pad_size, dtype=tf.float32)
        c_mask = tf.sequence_mask(self.c_len_ph, maxlen=self.context_pad_size, dtype=tf.float32)

        q_mask = tf.expand_dims(q_mask, -1)
        c_mask = tf.expand_dims(c_mask, -1)

        print(f"q_mask shape: {q_mask}")
        print(f"c mask shape: {c_mask}")

        q_embs = tf.multiply(q_embs, q_mask)
        c_embs = tf.multiply(c_embs, c_mask)

        print("Shapes after masking:")
        print(f"c_embs shape: {c_embs}")
        print(f"q_embs shape: {q_embs}")

        c_div = tf.expand_dims(tf.expand_dims(tf.cast(self.c_len_ph, dtype=tf.float32), -1), -1)
        q_div = tf.expand_dims(tf.cast(self.q_len_ph, dtype=tf.float32), -1)

        c_norm_emb = tf.reduce_sum(c_embs, axis=1, keepdims=True) / tf.sqrt(c_div)
        q_norm_emb = tf.reduce_sum(q_embs, axis=0, keepdims=True) / tf.sqrt(q_div)

        c_norm_emb = tf.squeeze(c_norm_emb)
        # q_norm_emb = tf.squeeze(q_norm_emb)

        print("Normalized shapes: ")
        print(f"c_norm shape: {c_norm_emb}")
        print(f"q_norm shape: {q_norm_emb}")

        dense_1 = tf.layers.dense(tf.reshape(c_norm_emb, [-1, self.emb_size]), units=self.hidden_size_dense_1,
                                  activation=tf.tanh,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
        dense_2 = tf.layers.dense(dense_1, units=self.hidden_size_dense_2, activation=tf.tanh,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
        dense_3 = tf.layers.dense(dense_2, units=self.hidden_size_dense_3, activation=tf.tanh,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
        dot_product = tf.squeeze(tf.matmul(q_norm_emb, tf.transpose(dense_3)))

        # if self.mode == 'infer':
        self.prob = tf.nn.softmax(logits=dot_product)

        labels = tf.sequence_mask(1, maxlen=batch_size, dtype=tf.uint8)

        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=dot_product, labels=labels)

        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def _init_placeholders(self):
        self.c_ph = tf.placeholder(shape=(None, None), dtype=tf.string, name='c_ph')
        self.q_ph = tf.placeholder(shape=(None, ), dtype=tf.string, name='q_ph')
        self.c_len_ph = tf.placeholder(shape=(None, ), dtype=tf.uint8,
                                       name='c_len_ph')
        self.q_len_ph = tf.placeholder(shape=(), dtype=tf.uint8,
                                       name='q_len_ph')

    def _init_optimizer(self):
        with tf.variable_scope('Optimizer'):
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                grads_and_vars = opt.compute_gradients(self.loss)
                self.train_op = opt.apply_gradients(grads_and_vars)

        # self.train_op = tf.train.MomentumOptimizer(self.learning_rate, 0).minimize(self.loss)

    def _build_feed_dict(self, query, chunk):
        # assert len(query) == len(chunk)
        q_len = len(query)
        c_len = [len(el) for el in chunk]
        q_pad = self._pad_sentences(query, self.question_pad_size)
        c_pad = self._pad_sentences(chunk, self.context_pad_size)
        feed_dict = {self.q_ph: q_pad, self.c_ph: c_pad, self.q_len_ph: q_len, self.c_len_ph: c_len}
        return feed_dict

    def train_on_batch(self, query: List[str], chunk: List[List[str]]):
        feed_dict = self._build_feed_dict(query, chunk)
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict)
        return loss

    def __call__(self, query: List[str], chunk: List[List[str]]):
        feed_dict = self._build_feed_dict(query, chunk)
        return self.sess.run(self.prob, feed_dict)

    @staticmethod
    def _pad_sentences(batch, limit):
        pad_batch = copy.deepcopy(batch)
        if isinstance(batch[0], list):
            for sentences in pad_batch:
                sentences += [''] * (limit - len(sentences))
        elif isinstance(batch[0], str):
            pad_batch += [''] * (limit - len(pad_batch))
        return pad_batch


q = ["Question 1?", "Sentence 2"]
c = [["Doc 1 sentence 1.", "Doc second sentence second."],
     ["Doc 2 sentence 1.", "Doc second sentence second.", "Third."]]
# y = [0, 1, 1]
encoder = BaseNeuralRankerEncoder()
prob = encoder(q, c)
print(prob)
_loss = encoder.train_on_batch(q, c)
print(_loss)
