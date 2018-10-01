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

from typing import Union, List
import copy
import tensorflow as tf
import tensorflow_hub as hub
from nltk.tokenize import sent_tokenize

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.tf_model import TFModel
from deeppavlov.core.common.log import get_logger

logger = get_logger(__name__)


@register('ranker_encoder')
class BasicNeuralQueryEncoder(TFModel):
    def __init__(self, save_path=None, load_path=None, **kwargs):
        self.hidden_size_dense_1 = 300
        self.hidden_size_dense_2 = 300
        self.hidden_size_dense_3 = 512
        self.learning_rate = 0.01
        self.question_pad_size = 3
        self.emb_size = 512
        self.n_epochs = 1000

        # self.mode = kwargs.get('mode', None)

        self.sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(
                allow_growth=True
            )))
        self._init_hub_modules()
        self._init_graph()
        self.sess.run(tf.global_variables_initializer())

        super().__init__(save_path=save_path, load_path=load_path)

        # load model if exists:

        if self.load_path is not None:
            self.load()

    def _init_hub_modules(self):
        self.embedder = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")

    def _init_graph(self):
        self._init_placeholders()

        q_embs = self.embedder(self.q_ph)

        q_embs = tf.stack(q_embs)

        # print(f"c_embs shape: {c_embs}")
        # print(f"q_embs shape: {q_embs}")

        q_mask = tf.sequence_mask(self.q_len_ph, maxlen=self.question_pad_size, dtype=tf.float32)

        q_mask = tf.expand_dims(q_mask, -1)

        # print(f"q_mask shape: {q_mask}")
        # print(f"c mask shape: {c_mask}")

        q_embs = tf.multiply(q_embs, q_mask)

        # print("Shapes after masking:")
        # print(f"c_embs shape: {c_embs}")
        # print(f"q_embs shape: {q_embs}")

        q_div = tf.expand_dims(tf.cast(self.q_len_ph, dtype=tf.float32), -1)

        self.q_norm_emb = tf.reduce_sum(q_embs, axis=0, keepdims=True) / tf.sqrt(q_div)

        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def _init_placeholders(self):
        self.q_ph = tf.placeholder(shape=(None, ), dtype=tf.string, name='q_ph')
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

    def _build_feed_dict(self, query):
        # assert len(query) == len(chunk)
        query = list(filter(lambda x: x != '', sent_tokenize(query)))
        q_len = len(query)
        q_pad = self._pad_sentences(query, self.question_pad_size)
        feed_dict = {self.q_ph: q_pad, self.q_len_ph: q_len}
        return feed_dict

    def train_on_batch(self, x, *args, **kwargs):
        pass

    def __call__(self, query: str):
        feed_dict = self._build_feed_dict(query)
        res = self.sess.run(self.q_norm_emb, feed_dict)
        return res

    @staticmethod
    def _pad_sentences(batch: Union[List[str], List[List[str]]], limit: int) -> Union[List[str], List[List[str]]]:
        """Pad a batch with empty string and cut batch if it exceeds the limit.

        Args:
            batch: data to be padded
            limit: pad size in integer value

        Returns:
            padded data

        """
        pad_batch = copy.deepcopy(batch)
        if isinstance(batch[0], list):
            for i in range(len(pad_batch)):
                if len(pad_batch[i]) >= limit:
                    pad_batch[i] = pad_batch[i][:limit]
                else:
                    pad_batch[i] += [''] * (limit - len(pad_batch[i]))
        elif isinstance(batch[0], str):
            if len(pad_batch) >= limit:
                pad_batch = pad_batch[:limit]
            else:
                pad_batch += [''] * (limit - len(pad_batch))
        return pad_batch

# q = ["Question 1?", "Sentence 2"]
# c = [["Doc 1 sentence 1.", "Doc second sentence second."],
#      ["Doc 2 sentence 1.", "Doc second sentence second.", "Third."]]
# # y = [0, 1, 1]
# encoder = BasicNeuralRankerEncoder()
# prob = encoder(q, c)
# print(prob)
# _loss = encoder.train_on_batch(q, c)
# print(_loss)
