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
class BasicNeuralContextEncoder(TFModel):
    def __init__(self, save_path=None, load_path=None, **kwargs):
        self.hidden_size_dense_1 = 300
        self.hidden_size_dense_2 = 300
        self.hidden_size_dense_3 = 512
        self.context_pad_size = 40
        self.emb_size = 512

        # self.mode = kwargs.get('mode', None)

        self.sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(
                allow_growth=True
            )))
        self._init_hub_modules()
        self._init_graph()
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

        super().__init__(save_path=save_path, load_path=load_path)

        # load model if exists:

        if self.load_path is not None:
            self.load()

    def _init_hub_modules(self):
        self.embedder = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")

    def _init_graph(self):
        self._init_placeholders()

        batch_size = tf.shape(self.c_ph)[0]
        c_embs = tf.reshape(self.c_ph, [-1])
        c_embs = self.embedder(c_embs)

        c_embs = tf.stack(c_embs)
        c_embs = tf.reshape(c_embs, shape=[batch_size, self.context_pad_size, self.emb_size])
        # q_embs = tf.reshape(q_embs, shape=[batch_size, self.question_pad_size, self.emb_size])

        # print(f"c_embs shape: {c_embs}")
        # print(f"q_embs shape: {q_embs}")

        c_mask = tf.sequence_mask(self.c_len_ph, maxlen=self.context_pad_size, dtype=tf.float32)

        c_mask = tf.expand_dims(c_mask, -1)

        # print(f"q_mask shape: {q_mask}")
        # print(f"c mask shape: {c_mask}")

        c_embs = tf.multiply(c_embs, c_mask)

        # print("Shapes after masking:")
        # print(f"c_embs shape: {c_embs}")
        # print(f"q_embs shape: {q_embs}")

        c_div = tf.expand_dims(tf.expand_dims(tf.cast(self.c_len_ph, dtype=tf.float32), -1), -1)

        c_norm_emb = tf.reduce_sum(c_embs, axis=1, keepdims=True) / tf.sqrt(c_div)

        c_norm_emb = tf.squeeze(c_norm_emb)
        # q_norm_emb = tf.squeeze(q_norm_emb)

        # print("Normalized shapes: ")
        # print(f"c_norm shape: {c_norm_emb}")
        # print(f"q_norm shape: {q_norm_emb}")

        dense_1 = tf.layers.dense(tf.reshape(c_norm_emb, [-1, self.emb_size]), units=self.hidden_size_dense_1,
                                  activation=tf.tanh,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
        dense_2 = tf.layers.dense(dense_1, units=self.hidden_size_dense_2, activation=tf.tanh,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.dense_3 = tf.layers.dense(dense_2, units=self.hidden_size_dense_3, activation=tf.tanh,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer())

    def _init_placeholders(self):
        self.c_ph = tf.placeholder(shape=(None, None), dtype=tf.string, name='c_ph')
        self.c_len_ph = tf.placeholder(shape=(None,), dtype=tf.uint8,
                                       name='c_len_ph')

    def _build_feed_dict(self, chunks):
        # assert len(query) == len(chunk)
        chunks = [list(filter(lambda x: x != '', sent_tokenize(ch))) for ch in chunks]
        c_len = [len(el) for el in chunks]
        c_pad = self._pad_sentences(chunks, self.context_pad_size)
        feed_dict = {self.c_ph: c_pad, self.c_len_ph: c_len}
        return feed_dict

    def __call__(self, chunks: List[str]):
        feed_dict = self._build_feed_dict(chunks)
        res = self.sess.run(self.dense_3, feed_dict)
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

    def train_on_batch(self, x: list, y: list):
        pass


# q = ["Question 1?", "Sentence 2"]
# c = [["Doc 1 sentence 1.", "Doc second sentence second."],
#      ["Doc 2 sentence 1.", "Doc second sentence second.", "Third."]]
# # y = [0, 1, 1]
# encoder = BasicNeuralRankerEncoder()
# prob = encoder(q, c)
# print(prob)
# _loss = encoder.train_on_batch(q, c)
# print(_loss)

# c = ['What a shiny day!', 'Hello world!']
# encoder = BasicNeuralContextEncoder(load_path='/media/olga/Data/projects/DeepPavlov/download/bnr/model')
# vectors = encoder(c)
# print(len(vectors))
# print(vectors[0].size)
