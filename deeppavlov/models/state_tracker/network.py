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
import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.models.tf_model import TFModel
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


@register("dst_network")
class StateTrackerNetwork(TFModel):
    """
    Parameters:
        hidden_size: RNN hidden layer size.
        num_slot_values:
        embedding_matrix:
        learning_rate: training learning rate.
        end_learning_rate: if set, learning rate starts from ``learning rate`` value and
            decays polynomially to the value of ``end_learning_rate``.
        decay_steps: number of steps for learning rate to decay.
        decay_power: power used to calculate learning rate decay for polynomial strategy.
        optimizer: one of tf.train.Optimizer subclasses as a string.
        **kwargs: parameters passed to a parent
                  :class:`~deeppavlov.core.models.tf_model.TFModel` class.
    """

    GRAPH_PARAMS = ['hidden_size', 'num_slot_values', 'num_tokens', 'embedding_size']

    def __init__(self,
                 hidden_size: int,
                 num_slot_values: int,
                 embedding_matrix: np.ndarray,
                 learning_rate: float,
                 end_learning_rate: float = None,
                 decay_steps: int = 1000,
                 decay_power: float = 1.,
                 optimizer: str = 'AdamOptimizer',
                 **kwargs) -> None:
        end_learning_rate = end_learning_rate or learning_rate
        # initializer embedding matrix
        self.emb_mat = embedding_matrix

        # specify model options
        self.opt = {
            'hidden_size': hidden_size,
            'num_slot_values': num_slot_values,
            'num_tokens': embedding_matrix.shape[0],
            'embedding_size': embedding_matrix.shape[1],
            'learning_rate': learning_rate,
            'end_learning_rate': end_learning_rate,
            'decay_steps': decay_steps,
            'decay_power': decay_power,
            'optimizer': optimizer
        }
        # initialize parameters
        self._init_params()
        # build computational graph
        self._build_graph()
        # initialize session
        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

        super().__init__(**kwargs)

        if tf.train.checkpoint_exists(str(self.save_path.resolve())):
            log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
            self.load()
        else:
            log.info("[initializing `{}` from scratch]".format(self.__class__.__name__))

    def _init_params(self):
        self.hidden_size = self.opt['hidden_size']
        self.num_slot_vals = self.opt['num_slot_values']
        self.num_tokens = self.opt['num_tokens']
        self.embedding_size = self.opt['embedding_size']
        self.learning_rate = self.opt['learning_rate']
        self.end_learning_rate = self.opt['end_learning_rate']
        self.decay_steps = self.opt['decay_steps']
        self.decay_power = self.opt['decay_power']

        self._optimizer = None
        if hasattr(tf.train, self.opt['optimizer']):
            self._optimizer = getattr(tf.train, self.opt['optimizer'])
        if not issubclass(self._optimizer, tf.train.Optimizer):
            raise ConfigError("`optimizer` parameter should be a name of"
                              " tf.train.Optimizer subclass")

    def _build_graph(self):

        self._add_placeholders()

        _logits, self._predictions = self._build_body()

        #_weights = tf.expand_dims(self._tgt_weights, -1)
        #_loss_tensor = \
        #    tf.losses.sparse_softmax_cross_entropy(logits=_logits,
        #                                           labels=self._decoder_outputs,
        #                                           weights=_weights,
        #                                           reduction=tf.losses.Reduction.NONE)
        # normalize loss by batch_size
        #self._loss = tf.reduce_sum(_loss_tensor) / tf.cast(self._batch_size, tf.float32)
        #self._train_op = \
        #    self.get_train_op(self._loss, self.learning_rate, clip_norm=10.)
        self._train_op, self._loss = _logits, self._predictions

    def _add_placeholders(self):
        # _utt_token_idx: [2, num_tokens]
        self._utt_token_idx = tf.placeholder(tf.int32,
                                             [2, None],
                                             name='utterance_token_indeces')
        # _utt_seq_length: [2]
        self._utt_seq_length = tf.placeholder(tf.int32,
                                              [2],
                                              name='utterance_lengths')
        # _utt_slot_idx: [num_slots, 2 * num_tokens]
        self._utt_slot_idx = tf.placeholder(tf.int32,
                                            [None, None],
                                            name='utterance_slot_token_indeces')
        self._num_slots = tf.shape(self._utt_slot_idx)[0]
        # _utt_act_feats: [1, 2 * num_dialog_actions]
        self._utt_act_feats = tf.placeholder(tf.float32,
                                             [1, None],
                                             name='utterance_action_features')
        # _slot_act_feats: [num_slots, 2 * num_dialog_actions]
        self._slot_act_feats = tf.placeholder(tf.float32,
                                              [None, None],
                                              name='slot_action_features')
        # _slot_val_act_feats: [num_slots, 2 * num_dialog_actions, max_num_slot_values]
        self._slot_val_act_feats = tf.placeholder(tf.float32,
                                                  [None, None, self.num_slot_vals],
                                                  name='slot_value_action_features')
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

        with tf.variable_scope("stacked_biGRU"):
            # _units1, _units2: [2, num_tokens, 2 * hidden_size]
            # _last_units: [2, 2 * hidden_size]
            (_units1, _units2), _last_units = \
                self._stacked_bi_gru(_utt_token_emb,
                                     [self.hidden_size, self.hidden_size],
                                     seq_length=self._utt_seq_length)
            # _utt_repr_tiled: [num_slots, 2 * 2 * hidden_size]
            _utt_repr_tiled = tf.tile(tf.reshape(_last_units, shape=(1, -1)),
                                      (self._num_slots, 1))
            # _token_repr: [2 * num_tokens, 4 * hidden_size]
            # TODO: lost 4 * hidden_size somewhere
            _token_repr = tf.reshape(tf.concat([_units1, _units2], axis=-1),
                                     shape=(-1, 4 * self.hidden_size))
            # _utt_slot_val_mask: [num_slots, 2 * num_tokens, num_values]
            _utt_slot_val_mask = tf.one_hot(self._utt_slot_idx - 1, self.num_slot_vals)
            # _slot_repr: [num_slots, num_values, 4 * hidden_size]
            _slot_repr = tf.tensordot(_utt_slot_val_mask, _token_repr, [[1], [0]])
        # return _utt_repr_tiled, _slot_repr
        return _utt_slot_val_mask, _slot_repr

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

    def __call__(self, u_utt_token_idx, u_utt_slot_idx_mat,
                 s_utt_token_idx, s_utt_slot_idx_mat,
                 prev_pred_mat, prob=False, *args, **kwargs):
        # print(f"u_utt_token_idx: {u_utt_token_idx}")
        # print(f"u_utt_slot_idx_mat: {u_utt_slot_idx_mat[0]}")
        # print(f"s_utt_token_idx: {s_utt_token_idx}")
        # print(f"s_utt_slot_idx_mat: {s_utt_slot_idx_mat[0]}")
        utt_token_idx, utt_seq_length = self._concat_and_pad([u_utt_token_idx,
                                                              s_utt_token_idx],
                                                             axis=0, pad_axis=1)
        utt_slot_idx, _ = self._concat_and_pad([u_utt_slot_idx_mat[0],
                                                s_utt_slot_idx_mat[0]],
                                               pad_length=max(utt_seq_length),
                                               axis=1, pad_axis=1)
        print(f"utt_token_idx: {utt_token_idx}")
        print(f"utt_slot_idx_mat: {utt_slot_idx}")
        predictions = self.sess.run(
            self._predictions,
            feed_dict={
                self._utt_token_idx: utt_token_idx,
                self._utt_seq_length: utt_seq_length,
                self._utt_slot_idx: utt_slot_idx,
                self._prev_pred: prev_pred_mat[0]
            }
        )
# TODO: implement infer probabilities
        if prob:
            raise NotImplementedError("Probs not available for now.")
        return predictions

    def train_on_batch(self, u_utt_token_idx, u_utt_slot_idx_mat,
                       s_utt_token_idx, s_utt_slot_idx_mat,
                       prev_pred_mat, true_state_mat, *args, **kwargs):
        utt_token_idx, utt_seq_length = self._concat_and_pad([u_utt_token_idx,
                                                              s_utt_token_idx],
                                                             axis=0, pad_axis=1)
        utt_slot_idx, _ = self._concat_and_pad([u_utt_slot_idx_mat[0],
                                                s_utt_slot_idx_mat[0]],
                                               pad_length=max(utt_seq_length),
                                               axis=1, pad_axis=1)
        _tr, loss_value = self.sess.run(
            [self._train_op, self._loss],
            feed_dict={
                self._utt_token_idx: utt_token_idx,
                self._utt_seq_length: utt_seq_length,
                self._utt_slot_idx: utt_slot_idx,
                self._prev_pred: prev_pred_mat[0],
                self._true_state: true_state_mat[0]
            }
        )
        print(f"utt_repr.shape: {_tr.shape}")
        print(f"where(utt_repr): {np.where(_tr)}")
        print(f"slot_repr.shape: {loss_value.shape}")
        return loss_value

    def load(self, *args, **kwargs):
        self.load_params()
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
