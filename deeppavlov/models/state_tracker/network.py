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

import json
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xav
import numpy as np
from time import time
from typing import Tuple

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.models.tf_model import EnhancedTFModel
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


@register("dst_network")
class StateTrackerNetwork(EnhancedTFModel):
    """
    Parameters:
        hidden_size: RNN hidden layer size.
        dense_sizes:
        num_slot_values:
        embedding_matrix:
        **kwargs: parameters passed to a parent
                  :class:`~deeppavlov.core.models.tf_model.TFModel` class.
    """

    GRAPH_PARAMS = ['hidden_size', 'dense_sizes', 'embedding_size',
                    'num_user_actions', 'num_system_actions', 'num_slot_values']

    def __init__(self,
                 hidden_size: int,
                 dense_sizes: Tuple[int, int],
                 num_slot_values: int,
                 num_user_actions: int,
                 num_system_actions: int,
                 embedding_matrix: np.ndarray,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        # initializer embedding matrix
        self.emb_mat = embedding_matrix

        # specify model options
        self.opt = {
            'hidden_size': hidden_size,
            'dense_sizes': dense_sizes,
            'num_slot_values': num_slot_values,
            'num_user_actions': num_user_actions,
            'num_system_actions': num_system_actions,
            'num_tokens': embedding_matrix.shape[0],
            'embedding_size': embedding_matrix.shape[1]
        }
        # initialize parameters
        self._init_params()
        # build computational graph
        self._build_graph()
        # initialize session
        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        if self.load_path and tf.train.checkpoint_exists(str(self.load_path.resolve())):
            log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
            self.load()
        else:
            log.info("[initializing `{}` from scratch]".format(self.__class__.__name__))
        self.batch_no = 0

    def _init_params(self):
        self.hidden_size = self.opt['hidden_size']
        self.dense_sizes = self.opt['dense_sizes']
        self.num_slot_vals = self.opt['num_slot_values']
        self.num_user_acts = self.opt['num_user_actions']
        self.num_sys_acts = self.opt['num_system_actions']
        self.num_tokens = self.opt['num_tokens']
        self.embedding_size = self.opt['embedding_size']

    def _build_graph(self):

        self._add_placeholders()

        # _logits: [num_slots, max_num_slot_values + 2]
        _logits = self._build_body()

        # _infmask_mask: [num_slots, max_num_slot_values + 2]
        _inf_mask = (self._slot_val_mask - 1) * tf.float32.max
        _logits = _logits * self._slot_val_mask + _inf_mask

        # _logits_exp, logits_exp_sum, _probs: [num_slots, max_num_slot_values + 2]
        # _logits_exp = tf.multiply(tf.exp(_logits), _logits_mask)
        # _logits_exp_sum = tf.reduce_sum(_logits_exp, axis=-1, keep_dims=True)
        # self._probs = _logits_exp / _logits_exp_sum

        # _prediction: [num_slots]
        self._logits = _logits
        self._prediction = tf.argmax(_logits, axis=-1, name='prediction')

        _loss_tensor = \
            tf.losses.softmax_cross_entropy(logits=_logits,
                                            onehot_labels=self._true_state,
                                            reduction=tf.losses.Reduction.NONE)
                                            # weights=self._slot_val_mask)
        # normalize loss by batch_size
        self._loss = tf.reduce_sum(_loss_tensor) / tf.cast(self._num_slots, tf.float32)

        tf.summary.scalar('loss', self._loss)
        self._summary = tf.summary.merge_all()
        self._train_writer = tf.summary.FileWriter(f"logs/train_{time()}", self.graph)
        self._test_writer = tf.summary.FileWriter(f"logs/test_{time()}")
        self._train_op = self.get_train_op(self._loss, clip_norm=2.)

    def _add_placeholders(self):
        # _utt_token_idx: [2, num_tokens]
        self._utt_token_idx = tf.placeholder(tf.int32,
                                             [2, None],
                                             name='utterance_token_indeces')
        # _utt_seq_length: [2]
        self._utt_seq_length = tf.placeholder(tf.int32,
                                              [2],
                                              name='utterance_lengths')
        # _utt_slot_tok_val_idx: [2, num_slots, num_tokens]
        self._utt_slot_tok_val_idx = tf.placeholder(tf.int32,
                                                    [2, None, None],
                                                    name='utterance_slot_value_indeces')
        self._num_slots = tf.shape(self._utt_slot_tok_val_idx)[1]
        # _u_utt_action_idx, _s_utt_action_idx: [utt_num_user_acts], [utt_num_sys_acts]
        self._u_utt_action_idx = tf.placeholder(tf.int32,
                                                [None],
                                                name='user_utterance_action_indeces')
        self._s_utt_action_idx = tf.placeholder(tf.int32,
                                                [None],
                                                name='system_utterance_action_indeces')
        # _u_utt_slot_action_mask, _s_utt_slot_action_mask: [num_slots, num_u/s_acts]
        self._u_utt_slot_action_mask = tf.placeholder(tf.float32,
                                                      [None, self.num_user_acts],
                                                      name='user_utter_slot_action_mask')
        self._s_utt_slot_action_mask = tf.placeholder(tf.float32,
                                                      [None, self.num_sys_acts],
                                                      name='sys_utter_slot_action_mask')
        # _slot_val_mask: [num_slots, max_num_slot_values + 2]
        self._slot_val_mask = tf.placeholder(tf.float32,
                                             [None, self.num_slot_vals + 2],
                                             name='slot_value_mask')
        # _prev_preds: [num_slots, max_num_slot_values + 2]
        self._prev_pred = tf.placeholder(tf.float32,
                                         [None, self.num_slot_vals + 2],
                                         name='previous_predictions')
        # _true_state: [num_slots, max_max_slot_values + 2]
        self._true_state = tf.placeholder(tf.float32,
                                          [None, self.num_slot_vals + 2],
                                          name='true_state')

        # TODO: try training embeddings
        # TODO: try weighting embeddings with tf-idf
        with tf.variable_scope("EmbeddingMatrix"):
            _emb_initializer = tf.constant_initializer(self.emb_mat)
            self._embedding = tf.get_variable("embedding",
                                              shape=self.emb_mat.shape,
                                              dtype=tf.float32,
                                              initializer=_emb_initializer,
                                              trainable=False)

    @staticmethod
    def _stacked_bi_gru(units, n_hidden_list, seq_length=None, name='rnn_layer'):
        outputs = []
        for n, n_hidden in enumerate(n_hidden_list):
            with tf.variable_scope(name + '_' + str(n)):
                forward_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
                backward_cell = tf.nn.rnn_cell.GRUCell(n_hidden)

                (rnn_output_fw, rnn_output_bw), (fw, bw) = \
                    tf.nn.bidirectional_dynamic_rnn(forward_cell,
                                                    backward_cell,
                                                    units,
                                                    dtype=tf.float32,
                                                    sequence_length=seq_length)
                units = tf.concat([rnn_output_fw, rnn_output_bw], axis=-1)
                outputs.append(units)
                last_units = tf.concat([fw, bw], axis=-1)
        return outputs, last_units

    def _build_body(self):
        # _utt_token_emb: [1, num_tokens, embedding_size]
        _utt_token_emb = tf.nn.embedding_lookup(self._embedding,
                                                self._utt_token_idx)

        with tf.variable_scope("action_features"):
            # _u_utt_action_mask, _s_utt_action_mask: [1,num_user_acts], [1,num_sys_acts]
            _u_utt_action_mask = tf.reduce_sum(tf.one_hot(self._u_utt_action_idx,
                                                          self.num_user_acts),
                                               axis=0, keep_dims=True)
            _s_utt_action_mask = tf.reduce_sum(tf.one_hot(self._s_utt_action_idx,
                                                          self.num_sys_acts),
                                               axis=0, keep_dims=True)
            # _utt_action_mask: [1, num_user_acts + num_sys_acts]
            _utt_action_mask = tf.concat([_u_utt_action_mask, _s_utt_action_mask], axis=1)
            # _utt_slot_action_mask: [num_slots, num_user_acts + num_sys_acts]
            _utt_slot_action_mask = tf.concat([self._u_utt_slot_action_mask,
                                               self._s_utt_slot_action_mask], axis=1)
            # _no_slot_action_mask: [1, num_user_acts + num_sys_acts]
            _no_slot_action_mask = tf.cast(tf.equal(tf.reduce_sum(_utt_slot_action_mask,
                                                                  axis=0, keep_dims=True),
                                                    0),
                                           tf.float32)
            # (1) _utt_no_slot_action_mask: [1, num_user_acts + num_sys_acts]
            _utt_no_slot_action_mask = _no_slot_action_mask * _utt_action_mask
            # (2) _utt_slot_action_mask: [num_slots, num_user_acts + num_sys_acts]
            _utt_slot_action_mask = _utt_slot_action_mask * _utt_action_mask
            # _utt_slot_val_mask: [2, num_slots, max_num_slot_vals]
            _utt_slot_val_mask = tf.reduce_max(tf.one_hot(self._utt_slot_tok_val_idx - 1,
                                                          self.num_slot_vals),
                                               axis=2)
            # _u_utt_slot_act_val_mask: [num_slots, max_num_slot_vals, num_user_acts]
            _u_utt_sl_act_val_mask = tf.tile(tf.expand_dims(self._u_utt_slot_action_mask,
                                                            axis=1),
                                             [1, self.num_slot_vals, 1]) *\
                tf.expand_dims(_utt_slot_val_mask[0], axis=-1)
            # _s_utt_slot_act_val_mask: [num_slots, max_num_slot_vals, num_sys_acts]
            _s_utt_sl_act_val_mask = tf.tile(tf.expand_dims(self._s_utt_slot_action_mask,
                                                            axis=1),
                                             [1, self.num_slot_vals, 1]) *\
                tf.expand_dims(_utt_slot_val_mask[1], axis=-1)
            # (3) _utt_slot_act_val_mask:
            #     [num_slots, max_num_slot_values, num_user_acts + num_sys_acts]
            _utt_slot_act_val_mask = tf.concat([_u_utt_sl_act_val_mask,
                                                _s_utt_sl_act_val_mask], axis=-1)

        with tf.variable_scope("stacked_biGRU"):
            # _units1, _units2: [2, num_tokens, 2 * hidden_size]
            # _last_units: [2, 2 * hidden_size]
            (_units1, _units2), _last_units = \
                self._stacked_bi_gru(_utt_token_emb,
                                     [self.hidden_size, self.hidden_size],
                                     seq_length=self._utt_seq_length)
            # _utt_birnn_repr: [2 * 2 * hidden_size]
            _utt_birnn_repr = tf.reshape(_last_units, shape=[-1])
            # _token_birnn_repr: [2 * num_tokens, 4 * hidden_size]
            _token_birnn_repr = tf.reshape(tf.concat([_units1, _units2], axis=-1),
                                           shape=(-1, 4 * self.hidden_size))
            # _utt_slot_tok_val_mask: [num_slots, 2 * num_tokens, max_num_slot_values]
            _utt_slot_tok_val_mask = tf.reshape(tf.one_hot(self._utt_slot_tok_val_idx - 1,
                                                           self.num_slot_vals),
                                                shape=(self._num_slots, -1,
                                                       self.num_slot_vals))
            # _slot_val_birnn_repr: [num_slots, max_num_slot_values, 4 * hidden_size]
            _slot_val_birnn_repr = tf.tensordot(_utt_slot_tok_val_mask, _token_birnn_repr,
                                                [[1], [0]])

        # _utt_repr: [1, 4 * hidden_size + num_user_acts + num_sys_acts]
        _utt_repr = tf.concat([tf.expand_dims(_utt_birnn_repr, axis=0),
                               _utt_no_slot_action_mask], axis=1)

        # _utt_repr_tiled: [num_slots, max_num_slot_values, 4*hidden_size + num_all_acts]
        _utt_repr_tiled = tf.tile(tf.reshape(_utt_repr, shape=[1, 1, -1]),
                                  (self._num_slots, self.num_slot_vals, 1))

        # _prev_pred_special_logits: [num_slots, 2]
        _prev_pred_special_logits = tf.slice(self._prev_pred, [0, 0], [-1, 2])
        # _slot_repr: [num_slots, 2 + num_all_acts]
        _slot_repr = tf.concat([_prev_pred_special_logits, _utt_slot_action_mask],
                               axis=1)
        # _slot_repr_tiled: [num_slots, max_num_slot_values, 2 + num_all_acts]
        _slot_repr_tiled = tf.tile(tf.expand_dims(_slot_repr, axis=1),
                                   (1, self.num_slot_vals, 1))

        # _prev_pred_wo_special_logits: [num_slots, max_num_slot_values]
        _prev_pred_wo_special_logits = tf.slice(self._prev_pred, [0, 2], [-1, -1])
        # _slot_val_repr: [num_slots, max_num_slot_values, 4*hidden_size + num_all_acts+1]
        _slot_val_repr = tf.concat([_slot_val_birnn_repr,
                                    tf.expand_dims(_prev_pred_wo_special_logits, -1),
                                    _utt_slot_act_val_mask],
                                   axis=-1)

        with tf.variable_scope("CandidateScorer"):
            # _cand_feats:
            # [num_slots, max_num_slot_values, 8 * hidden_size + 3 * num_all_acts + 3]
            _cand_feats = tf.concat([_utt_repr_tiled, _slot_repr_tiled, _slot_val_repr],
                                    -1)
            # _proj: [num_slots, max_num_slot_values, dense_size0]
            _proj = tf.layers.dense(_cand_feats, self.dense_sizes[0],
                                    activation=tf.nn.sigmoid,
                                    kernel_initializer=xav())
            # _logits: [num_slots, max_num_slot_values]
            _logits = tf.squeeze(tf.layers.dense(_proj, 1, kernel_initializer=xav()),
                                 axis=[-1])

            with tf.variable_scope("DontcareScorer"):
                # _utt_repr_slot_tiled: [num_slots, 4 * hidden_size + num_all_acts]
                _utt_repr_slot_tiled = tf.tile(tf.reshape(_utt_repr, [1, -1]),
                                               (self._num_slots, 1))
                # _dontcare_feats: [num_slots, 4 * hidden_size + 2 * num_all_acts + 2]
                _dontcare_feats = tf.concat([_utt_repr_slot_tiled, _slot_repr], -1)
                # _dontcare_proj: [num_slots, dense_size1]
                _dontcare_proj = tf.layers.dense(_dontcare_feats, self.dense_sizes[1],
                                                 activation=tf.nn.sigmoid,
                                                 kernel_initializer=xav())
                # _dontcare_logit: [num_slots, 1]
                _dontcare_logit = tf.layers.dense(_dontcare_proj, 1,
                                                  kernel_initializer=xav())

        _null_bias = tf.get_variable("null_bias", shape=(1, 1), dtype=tf.float32,
                                     trainable=True)
        # _null_bias_tiled: [num_slots, 1]
        _null_bias_tiled = tf.tile(_null_bias, (self._num_slots, 1))
        # _logits: [num_slots, max_num_slot_values + 2]
        _logits = tf.concat([_null_bias_tiled, _dontcare_logit, _logits], -1)
        return _logits

    @staticmethod
    def _concat_and_pad(arrays, axis, pad_axis, pad_value=0, pad_length=None):
        """Concatenate arrays along `axis`, reshape and padd along `pad_axis`."""
        # ensure that arrays are np.array objects
        if not isinstance(arrays[0], np.ndarray):
            arrays = [np.array(arr) for arr in arrays]
        # calculate sequence length along padding axis-of-interest
        seq_length = [arr.shape[pad_axis] for arr in arrays]
        # calculate maximum of sequence lengths and check pad_length
        max_seq_length = max(seq_length)
        if (pad_length is not None) and (max_seq_length > pad_length):
            raise RuntimeError(f"padding length {pad_length} is too small.")
        if pad_length is None:
            pad_length = max_seq_length
        # take memory for concatenation result
        result_shape = list(arrays[0].shape)
        result_shape[pad_axis] = pad_length
        result_shape[axis] *= len(arrays)
        result_dtype = max(arr.dtype for arr in arrays)
        result = np.ones(result_shape, dtype=result_dtype) * pad_value
        # put arrays into result using advanced slicing indeces
        slicing = [slice(None)] * len(result_shape)
        for i, (arr, arr_len) in enumerate(zip(arrays, seq_length)):
            if axis != pad_axis:
                slicing[axis] = i
                slicing[pad_axis] = slice(arr_len)
            else:
                slicing[axis] = slice(i * pad_length, i * pad_length + arr_len)
            result[tuple(slicing)] = arr
        return result, seq_length

    def __call__(self, u_utt_token_idx, u_utt_slot_tok_val_idx, u_act_idx,
                 u_slot_act_mask,
                 s_utt_token_idx, s_utt_slot_tok_val_idx, s_act_idx, s_slot_act_mask,
                 slot_val_mask, prev_pred_mat, prob=False, **kwargs):
        # print(f"u_utt_token_idx: {u_utt_token_idx}")
        # print(f"s_utt_token_idx: {s_utt_token_idx}")
        utt_token_idx, utt_seq_length = self._concat_and_pad([u_utt_token_idx,
                                                              s_utt_token_idx],
                                                             axis=0, pad_axis=1)
        u_utt_slot_tok_val_idx = np.expand_dims(u_utt_slot_tok_val_idx[0], axis=0)
        s_utt_slot_tok_val_idx = np.expand_dims(s_utt_slot_tok_val_idx[0], axis=0)
        # print(f"u_utt_slot_tok_val_idx: {u_utt_slot_tok_val_idx}")
        # print(f"s_utt_slot_tok_val_idx: {s_utt_slot_tok_val_idx}")
        utt_slot_tok_val_idx, _ = self._concat_and_pad([u_utt_slot_tok_val_idx,
                                                        s_utt_slot_tok_val_idx],
                                                       pad_length=max(utt_seq_length),
                                                       axis=0, pad_axis=2)
        # print(f"utt_token_idx: {utt_token_idx}")
        # print(f"utt_slot_idx_mat: {utt_slot_idx}")
        # prediction, logits = self.sess.run(
        #    [self._prediction, self._logits],
        prediction = self.sess.run(
            self._prediction,
            feed_dict={
                self._u_utt_action_idx: u_act_idx,
                self._u_utt_slot_action_mask: u_slot_act_mask[0],
                self._s_utt_action_idx: s_act_idx,
                self._s_utt_slot_action_mask: s_slot_act_mask[0],
                self._utt_token_idx: utt_token_idx,
                self._utt_seq_length: utt_seq_length,
                self._utt_slot_tok_val_idx: utt_slot_tok_val_idx,
                self._slot_val_mask: slot_val_mask[0],
                self._prev_pred: prev_pred_mat[0]
            }
        )
# TODO: implement infer probabilities
        if prob:
            raise NotImplementedError("Probs not available for now.")
        # logits = np.reshape(logits, [1, -1])
        # return prediction, logits
        return prediction

    def train_on_batch(self, u_utt_token_idx, u_utt_slot_tok_val_idx, u_act_idx,
                       u_slot_act_mask,
                       s_utt_token_idx, s_utt_slot_tok_val_idx, s_act_idx,
                       s_slot_act_mask,
                       slot_val_mask, prev_pred_mat, true_state_mat, **kwargs):
        utt_token_idx, utt_seq_length = self._concat_and_pad([u_utt_token_idx,
                                                              s_utt_token_idx],
                                                             axis=0, pad_axis=1)
        u_utt_slot_tok_val_idx = np.expand_dims(u_utt_slot_tok_val_idx[0], axis=0)
        s_utt_slot_tok_val_idx = np.expand_dims(s_utt_slot_tok_val_idx[0], axis=0)
        utt_slot_tok_val_idx, _ = self._concat_and_pad([u_utt_slot_tok_val_idx,
                                                        s_utt_slot_tok_val_idx],
                                                       pad_length=max(utt_seq_length),
                                                       axis=0, pad_axis=2)
        _tr, loss_value, summary = self.sess.run(
            [self._train_op, self._loss, self._summary],
            feed_dict={
                self.get_learning_rate_ph(): self.get_learning_rate(),
                self.get_momentum_ph(): self.get_momentum(),
                self._u_utt_action_idx: u_act_idx,
                self._u_utt_slot_action_mask: u_slot_act_mask[0],
                self._s_utt_action_idx: s_act_idx,
                self._s_utt_slot_action_mask: s_slot_act_mask[0],
                self._utt_token_idx: utt_token_idx,
                self._utt_seq_length: utt_seq_length,
                self._utt_slot_tok_val_idx: utt_slot_tok_val_idx,
                self._prev_pred: prev_pred_mat[0],
                self._slot_val_mask: slot_val_mask[0],
                self._true_state: true_state_mat[0]
            }
        )
        self.batch_no += 1
        self._train_writer.add_summary(summary, self.batch_no)
        # print(f"_tr.shape: {_tr.shape}")
        # print(f"_tr: {_tr}")
        # print(f"where(_tr): {np.where(_tr)}")
        # print(f"loss_value.shape: {loss_value.shape}")
        # print(f"loss_value: {loss_value}")
        report = {'loss': loss_value, 'learning_rate': self.get_learning_rate()}
        if self.get_momentum():
            report['momentum'] = self.get_momentum()
        return report

    def load(self, *args, **kwargs):
        self.load_params()
        kwargs['exclude_scopes'] = ('Optimizer', 'EmbeddingMatrix')
        super().load(*args, **kwargs)

    def load_params(self):
        path = str(self.load_path.with_suffix('.json').resolve())
        log.info('[loading parameters from {}]'.format(path))
        with open(path, 'r', encoding='utf8') as fp:
            params = json.load(fp)
        for p in self.GRAPH_PARAMS:
            if self.opt.get(p) != params.get(p):
                raise ConfigError("`{}` parameter must be equal to saved model"
                                  " parameter value `{}`, but is equal to `{}`"
                                  .format(p, params.get(p), self.opt.get(p)))

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        self.save_params()

    def save_params(self):
        path = str(self.save_path.with_suffix('.json').resolve())
        log.info('[saving parameters to {}]'.format(path))
        with open(path, 'w', encoding='utf8') as fp:
            json.dump(self.opt, fp)

    def shutdown(self):
        self.sess.close()
