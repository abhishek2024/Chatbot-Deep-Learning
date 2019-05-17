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

import sys
from logging import getLogger
from typing import Iterator, List, Union, Optional

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from overrides import overrides
from bert_dp.modeling import BertConfig, BertModel
from bert_dp.optimization import AdamWeightDecayOptimizer
from bert_dp.preprocessing import InputFeatures

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import zero_pad, chunk_generator
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.tf_backend import TfModelMeta

log = getLogger(__name__)


@register('bert_embedder')
class BertEmbedder(Component, metaclass=TfModelMeta):
    """Token and Sentence Embeddings from Bert model.

    Args:
        bert_config_file: path to Bert configuration file
        attention_probs_keep_prob: keep_prob for Bert self-attention layers
        hidden_keep_prob: keep_prob for Bert hidden layers
        pretrained_bert: pretrained Bert checkpoint
        pad_zero: whether to use pad samples or not
        mean: whether to return one mean embedding vector per sample
    """
    def __init__(self, bert_config_file,
                 attention_probs_keep_prob=None, hidden_keep_prob=None,
                 pretrained_bert=None, pad_zero: bool = False, mean: bool = False, **kwargs) -> None:

        self.tok2emb = {}
        self.pad_zero = pad_zero
        self.mean = mean
        self.dim = None

        super().__init__(**kwargs)

        self.bert_config = BertConfig.from_json_file(str(expand_path(bert_config_file)))

        if attention_probs_keep_prob is not None:
            self.bert_config.attention_probs_dropout_prob = 1.0 - attention_probs_keep_prob
        if hidden_keep_prob is not None:
            self.bert_config.hidden_dropout_prob = 1.0 - hidden_keep_prob

        self.sess_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.sess_config)

        self._init_graph()

        self.sess.run(tf.global_variables_initializer())

        if pretrained_bert is not None:
            pretrained_bert = str(expand_path(pretrained_bert))

        if tf.train.checkpoint_exists(pretrained_bert) \
                and not tf.train.checkpoint_exists(str(self.load_path.resolve())):
            log.info('[initializing model with Bert from {}]'.format(pretrained_bert))
            # Exclude optimizer and classification variables from saved variables
            var_list = self._get_saveable_variables(
                exclude_scopes=('Optimizer', 'learning_rate', 'momentum', 'output_weights', 'output_bias'))
            saver = tf.train.Saver(var_list)
            saver.restore(self.sess, pretrained_bert)

        if self.load_path is not None:
            self.load()

    def _init_graph(self):
        self._init_placeholders()

        self.bert = BertModel(config=self.bert_config,
                              is_training=self.is_train_ph,
                              input_ids=self.input_ids_ph,
                              input_mask=self.input_masks_ph,
                              token_type_ids=self.token_types_ph,
                              use_one_hot_embeddings=False,
                              )
        if self.mean:
            # get_sequence_output returns the encoded sequence tensor of shape
            # [batch_size, seq_length, hidden_size] that is exactly tokens embeddings
            self.embeddings = self.bert.get_sequence_output()
        else:
            # get_pooled_output returns the encoded tensor of shape
            # [batch_size, hidden_size] that is exactly sentence embeddings
            self.embeddings = self.bert.get_pooled_output()

    def _init_placeholders(self):
        self.input_ids_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='ids_ph')
        self.input_masks_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='masks_ph')
        self.token_types_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='token_types_ph')

        if self.mean:
            self.embeddings = tf.placeholder(shape=(None, None), dtype=tf.float32, name='embeddings')
        else:
            self.embeddings = tf.placeholder(shape=(None, None, None), dtype=tf.float32, name='embeddings')

        self.is_train_ph = tf.placeholder_with_default(False, shape=[], name='is_train_ph')

    def _build_feed_dict(self, input_ids, input_masks, token_types):
        feed_dict = {
            self.input_ids_ph: input_ids,
            self.input_masks_ph: input_masks,
            self.token_types_ph: token_types,
        }
        return feed_dict

    @overrides
    def __call__(self, features: List[InputFeatures], *args, **kwargs) -> List[Union[list, np.ndarray]]:
        """
        Embed sentences from batch

        Args:
            batch: list of tokenized text samples
            mean: whether to return mean embedding of tokens per sample

        Returns:
            embedded batch
        """
        input_ids = [f.input_ids for f in features]
        input_masks = [f.input_mask for f in features]
        input_type_ids = [f.input_type_ids for f in features]

        feed_dict = self._build_feed_dict(input_ids, input_masks, input_type_ids)

        embeddings = self.sess.run(self.embeddings, feed_dict=feed_dict)

        if self.pad_zero:
            batch = zero_pad(embeddings)
        return batch

    def destroy(self):
        if hasattr(self, 'sess'):
            for k in list(self.sess.graph.get_all_collection_keys()):
                self.sess.graph.clear_collection(k)
        super().destroy()
