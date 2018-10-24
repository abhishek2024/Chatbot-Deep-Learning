"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow as tf
import tensorflow_hub as tfhub
import numpy as np

from deeppavlov.core.layers.tf_layers import cudnn_bi_gru, variational_dropout


class CudnnGRU:
    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, share_layers=False):
        self.num_layers = num_layers
        self.share_layers = share_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units

            init_fw = tf.Variable(tf.zeros([num_units]))
            init_fw = tf.expand_dims(tf.tile(tf.expand_dims(init_fw, axis=0), [batch_size, 1]), axis=0)
            init_bw = tf.Variable(tf.zeros([num_units]))
            init_bw = tf.expand_dims(tf.tile(tf.expand_dims(init_bw, axis=0), [batch_size, 1]), axis=0)

            mask_fw = tf.nn.dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                                    keep_prob=keep_prob)
            mask_bw = tf.nn.dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                                    keep_prob=keep_prob)

            gru_fw = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=1, num_units=num_units)
            gru_bw = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=1, num_units=num_units)
            self.grus.append((gru_fw, gru_bw,))

            self.inits.append((init_fw, init_bw,))
            self.dropout_mask.append((mask_fw, mask_bw,))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        outputs = [tf.transpose(inputs, [1, 0, 2])]
        for layer in range(self.num_layers):
            gru_fw, gru_bw = self.grus[layer]
            init_fw, init_bw = self.inits[layer]
            mask_fw, mask_bw = self.dropout_mask[layer]
            with tf.variable_scope('fw_{}'.format(layer if not self.share_layers else 0), reuse=tf.AUTO_REUSE):
                out_fw, _ = gru_fw(outputs[-1] * mask_fw, (init_fw, ))
            with tf.variable_scope('bw_{}'.format(layer if not self.share_layers else 0), reuse=tf.AUTO_REUSE):
                inputs_bw = tf.reverse_sequence(
                    outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                out_bw, _ = gru_bw(inputs_bw, (init_bw, ))
                out_bw = tf.reverse_sequence(
                    out_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
            outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        res = tf.transpose(res, [1, 0, 2])
        return res


class CudnnGRULegacy:
    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, share_layers=None):
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            gru_fw = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=1, num_units=num_units)
            gru_bw = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=1, num_units=num_units)

            init_fw = tf.Variable(tf.zeros([num_units]))
            init_fw = tf.expand_dims(tf.tile(tf.expand_dims(init_fw, axis=0), [batch_size, 1]), axis=0)
            init_bw = tf.Variable(tf.zeros([num_units]))
            init_bw = tf.expand_dims(tf.tile(tf.expand_dims(init_bw, axis=0), [batch_size, 1]), axis=0)

            mask_fw = tf.nn.dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                                    keep_prob=keep_prob)
            mask_bw = tf.nn.dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                                    keep_prob=keep_prob)

            self.grus.append((gru_fw, gru_bw,))
            self.inits.append((init_fw, init_bw,))
            self.dropout_mask.append((mask_fw, mask_bw,))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        outputs = [tf.transpose(inputs, [1, 0, 2])]
        for layer in range(self.num_layers):
            gru_fw, gru_bw = self.grus[layer]
            init_fw, init_bw = self.inits[layer]
            mask_fw, mask_bw = self.dropout_mask[layer]
            with tf.variable_scope('fw', reuse=tf.AUTO_REUSE):
                out_fw, _ = gru_fw(outputs[-1] * mask_fw, (init_fw, ))
            with tf.variable_scope('bw', reuse=tf.AUTO_REUSE):
                inputs_bw = tf.reverse_sequence(
                    outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                out_bw, _ = gru_bw(inputs_bw, (init_bw, ))
                out_bw = tf.reverse_sequence(
                    out_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
            outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        res = tf.transpose(res, [1, 0, 2])
        return res


class CudnnCompatibleGRU:
    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, share_layers=False):
        self.num_layers = num_layers
        self.share_layers = share_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units

            init_fw = tf.Variable(tf.zeros([num_units]))
            init_fw = tf.expand_dims(tf.tile(tf.expand_dims(init_fw, axis=0), [batch_size, 1]), axis=0)
            init_bw = tf.Variable(tf.zeros([num_units]))
            init_bw = tf.expand_dims(tf.tile(tf.expand_dims(init_bw, axis=0), [batch_size, 1]), axis=0)

            mask_fw = tf.nn.dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                                    keep_prob=keep_prob)
            mask_bw = tf.nn.dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                                    keep_prob=keep_prob)

            gru_fw = tf.nn.rnn_cell.MultiRNNCell([
                tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(num_units=num_units)])

            gru_bw = tf.nn.rnn_cell.MultiRNNCell([
                tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(num_units=num_units)])

            self.grus.append((gru_fw, gru_bw,))

            self.inits.append((init_fw, init_bw,))
            self.dropout_mask.append((mask_fw, mask_bw,))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        outputs = [tf.transpose(inputs, [1, 0, 2])]
        for layer in range(self.num_layers):
            gru_fw, gru_bw = self.grus[layer]
            init_fw, init_bw = self.inits[layer]
            mask_fw, mask_bw = self.dropout_mask[layer]
            with tf.variable_scope('fw_{}'.format(layer if not self.share_layers else 0), reuse=tf.AUTO_REUSE):
                with tf.variable_scope('cudnn_gru', reuse=tf.AUTO_REUSE):
                    out_fw, _ = tf.nn.dynamic_rnn(cell=gru_fw, inputs=outputs[-1] * mask_fw, time_major=True,
                                                  initial_state=tuple(tf.unstack(init_fw, axis=0)))

            with tf.variable_scope('bw_{}'.format(layer if not self.share_layers else 0), reuse=tf.AUTO_REUSE):
                with tf.variable_scope('cudnn_gru', reuse=tf.AUTO_REUSE):
                    inputs_bw = tf.reverse_sequence(
                        outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                    out_bw, _ = tf.nn.dynamic_rnn(cell=gru_bw, inputs=inputs_bw, time_major=True,
                                                  initial_state=tuple(tf.unstack(init_bw, axis=0)))
                    out_bw = tf.reverse_sequence(
                        out_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)

            outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        res = tf.transpose(res, [1, 0, 2])
        return res


class PtrNet:
    def __init__(self, cell_size, keep_prob=1.0, scope="ptr_net"):
        self.gru = tf.nn.rnn_cell.GRUCell(cell_size)
        self.scope = scope
        self.keep_prob = keep_prob

    def __call__(self, init, match, att_hidden_size, mask):
        with tf.variable_scope(self.scope):
            BS, ML, MH = tf.unstack(tf.shape(match))
            BS, IH = tf.unstack(tf.shape(init))
            match_do = tf.nn.dropout(match, keep_prob=self.keep_prob, noise_shape=[BS, 1, MH])
            dropout_mask = tf.nn.dropout(tf.ones([BS, IH], dtype=tf.float32), keep_prob=self.keep_prob)
            inp, logits1 = attention(match_do, init * dropout_mask, att_hidden_size, mask)
            inp_do = tf.nn.dropout(inp, keep_prob=self.keep_prob)
            _, state = self.gru(inp_do, init)
            tf.get_variable_scope().reuse_variables()
            _, logits2 = attention(match_do, state * dropout_mask, att_hidden_size, mask)
            return logits1, logits2


def dot_attention(inputs, memory, mask, att_size, keep_prob=1.0,
                  use_gate=True, drop_diag=False, use_transpose_att=False, inputs_mask=None, concat_inputs=True,
                  scope="dot_attention"):
    """Computes attention vector for each item in inputs:
       attention vector is a weighted sum of memory items.
       Dot product between input and memory vector is used as similarity measure.

       Gate mechanism is applied to attention vectors to produce output.

    Args:
        inputs: Tensor [batch_size x input_len x feature_size]
        memory: Tensor [batch_size x memory_len x feature_size]
        mask: inputs mask
        att_size: hidden size of attention
        keep_prob: dropout keep_prob
        use_gate: use output sigmoid gate or not
        drop_diag: set attention weights to zero on diagonal (could be useful for self-attention)
        use_transpose_att: additionally compute memory to inputs attention
        scope: variable scope to use

    Returns:
        attention vectors [batch_size x input_len x (feature_size + feature_size)]

    """
    with tf.variable_scope(scope):
        BS, IL, IH = tf.unstack(tf.shape(inputs))
        BS, ML, MH = tf.unstack(tf.shape(memory))

        d_inputs = variational_dropout(inputs, keep_prob=keep_prob)
        d_memory = variational_dropout(memory, keep_prob=keep_prob)

        with tf.variable_scope("attention"):
            inputs_att = tf.layers.dense(d_inputs, att_size, use_bias=False,
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                         kernel_regularizer=tf.nn.l2_loss)
            memory_att = tf.layers.dense(d_memory, att_size, use_bias=False,
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                         kernel_regularizer=tf.nn.l2_loss)
            logits = tf.matmul(inputs_att, tf.transpose(memory_att, [0, 2, 1])) / (att_size ** 0.5)

            mask = tf.tile(tf.expand_dims(mask, axis=1), [1, IL, 1])

            if drop_diag:
                # set mask values to zero on diag
                mask = tf.cast(mask, tf.float32) * (1 - tf.diag(tf.ones(shape=(IL,), dtype=tf.float32)))

            att_weights = tf.nn.softmax(softmax_mask(logits, mask))
            outputs = tf.matmul(att_weights, memory)

            if concat_inputs:
                res = tf.concat([inputs, outputs], axis=2)
            else:
                res = outputs

            # like in QA-NET and DCN: S * SS_T * C
            if use_transpose_att:
                if inputs_mask is None:
                    inputs_mask = tf.ones(shape=(BS, IL), dtype=tf.float32)

                transpose_mask = tf.transpose(tf.tile(tf.expand_dims(inputs_mask, axis=1), [1, ML, 1]), [0, 2, 1])
                transpose_att_weights = tf.nn.softmax(softmax_mask(logits, transpose_mask), axis=1)

                S_SS_T = tf.matmul(att_weights, transpose_att_weights, transpose_b=True)

                transpose_outputs = tf.matmul(S_SS_T, inputs)

                res = tf.concat([res, inputs * outputs, inputs * transpose_outputs], axis=2)

        if use_gate:
            with tf.variable_scope("gate"):
                dim = res.get_shape().as_list()[-1]
                d_res = tf.nn.dropout(res, keep_prob=keep_prob, noise_shape=[BS, 1, dim])
                gate = tf.layers.dense(d_res, dim, use_bias=False,
                                       activation=tf.nn.sigmoid,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       kernel_regularizer=tf.nn.l2_loss)
                return res * gate

        return res


def dot_reattention(inputs, memory, memory_mask, inputs_mask, att_size, E=None, B=None, gamma_init=3, keep_prob=1.0,
                    drop_diag=False, concat_inputs=False, scope="dot_reattention"):
    # check reinforced mnemonic reader paper for more info about E, B and re-attention
    with tf.variable_scope(scope):
        BS, IL, IH = tf.unstack(tf.shape(inputs))
        BS, ML, MH = tf.unstack(tf.shape(memory))

        d_inputs = variational_dropout(inputs, keep_prob=keep_prob)
        d_memory = variational_dropout(memory, keep_prob=keep_prob)

        with tf.variable_scope("attention"):
            inputs_att = tf.layers.dense(d_inputs, att_size, use_bias=False,
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                         kernel_regularizer=tf.nn.l2_loss)
            memory_att = tf.layers.dense(d_memory, att_size, use_bias=False,
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                         kernel_regularizer=tf.nn.l2_loss)
            # BS x IL x ML
            logits = tf.matmul(inputs_att, tf.transpose(memory_att, [0, 2, 1])) / (att_size ** 0.5)

            if E is not None and B is not None:
                gamma = tf.Variable(gamma_init, dtype=tf.float32, trainable=True, name='gamma')
                E_softmax = tf.nn.softmax(softmax_mask(E,
                                                       tf.tile(tf.expand_dims(inputs_mask, axis=-1), [1, 1, ML])),
                                          axis=1)

                B_softmax = tf.nn.softmax(softmax_mask(B,
                                                       tf.tile(tf.expand_dims(inputs_mask, axis=1), [1, IL, 1])),
                                          axis=-1)

                logits = logits + gamma * tf.matmul(B_softmax, E_softmax)

            memory_mask = tf.tile(tf.expand_dims(memory_mask, axis=1), [1, IL, 1])
            if drop_diag:
                # set mask values to zero on diag
                memory_mask = tf.cast(memory_mask, tf.float32) * (1 - tf.diag(tf.ones(shape=(IL,), dtype=tf.float32)))

            att_weights = tf.nn.softmax(softmax_mask(logits, memory_mask))
            outputs = tf.matmul(att_weights, memory)

            if concat_inputs:
                res = tf.concat([inputs, outputs], axis=2)
            else:
                res = outputs

        return res, logits


def simple_attention(memory, att_size, mask, keep_prob=1.0, scope="simple_attention"):
    """Simple attention without any conditions.
        a_i = v^T * tanh(W * m_i + b)

       Computes weighted sum of memory elements.
    """
    with tf.variable_scope(scope):
        BS, ML, MH = tf.unstack(tf.shape(memory))
        memory_do = tf.nn.dropout(memory, keep_prob=keep_prob, noise_shape=[BS, 1, MH])
        logits = tf.layers.dense(
            tf.layers.dense(memory_do, att_size,
                            activation=tf.nn.tanh,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.nn.l2_loss),
            1, use_bias=False)
        logits = softmax_mask(tf.squeeze(logits, [2]), mask)
        att_weights = tf.expand_dims(tf.nn.softmax(logits), axis=2)
        res = tf.reduce_sum(att_weights * memory, axis=1)
        return res


def attention(inputs, state, att_size, mask, use_combinations=False, scope="attention", reuse=False):
    """Computes weighted sum of inputs conditioned on state

        Additive form of attention:
        a_i = v^T * tanh(W * [state, m_i] + b)
    """
    with tf.variable_scope(scope, reuse=reuse):
        state_tiled = tf.tile(tf.expand_dims(state, axis=1), [1, tf.shape(inputs)[1], 1])
        u = tf.concat([state_tiled, inputs], axis=2)
        if use_combinations:
            u = tf.concat([u, state_tiled, inputs * state_tiled, inputs - state_tiled], axis=2)
        logits = tf.layers.dense(
            tf.layers.dense(u, att_size,
                            activation=tf.nn.tanh,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.nn.l2_loss),
            1, use_bias=False)
        logits = softmax_mask(tf.squeeze(logits, [2]), mask)
        att_weights = tf.expand_dims(tf.nn.softmax(logits), axis=2)
        res = tf.reduce_sum(att_weights * inputs, axis=1)
        return res, logits


def mult_attention(inputs, state, mask, scope="attention", reuse=False):
    """Computes weighted sum of inputs conditioned on state

        Multiplicative form of attention:
        a_i = state * W * m_i
    """
    with tf.variable_scope(scope, reuse=reuse):
        logits = tf.matmul(tf.expand_dims(state, axis=1),
                           tf.layers.dense(inputs, units=state.get_shape()[-1], use_bias=False, reuse=reuse),
                           transpose_b=True)
        logits = softmax_mask(tf.squeeze(logits, axis=1), mask)
        att_weights = tf.expand_dims(tf.nn.softmax(logits), axis=2)
        res = tf.reduce_sum(att_weights * inputs, axis=1)
        return res, logits


def softmax_mask(val, mask):
    INF = 1e30
    return -INF * (1 - tf.cast(mask, tf.float32)) + val


def embedding_layer(ids, emb_mat_init=None, vocab_size=None, emb_dim=None,
                    trainable=False, regularizer=None, scope='embedding_layer'):

    if emb_mat_init is not None:
        init_mat = emb_mat_init.astype(np.float32)
    else:
        init_mat = np.random.randn(vocab_size, emb_dim).astype(np.float32) / np.sqrt(emb_dim)

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        tok_emb_mat = tf.get_variable('emb_mat', initializer=init_mat,
                                      dtype=tf.float32, trainable=trainable, regularizer=regularizer)
        embedded_tokens = tf.nn.embedding_lookup(tok_emb_mat, ids)

    return embedded_tokens


def character_embedding_layer(ids, char_encoder_hidden_size, keep_prob, emb_mat_init=None, vocab_size=None, emb_dim=None,
                              trainable_emb_mat=False, regularizer=None, transform_char_emb=0,
                              scope='character_embedding_layer', reuse=False):
    # padding is a char with id == 0
    with tf.variable_scope(scope, reuse=reuse):
        c_emb = embedding_layer(ids, emb_mat_init, vocab_size, emb_dim, trainable=trainable_emb_mat,
                                         regularizer=regularizer)
        token_dim = tf.shape(c_emb)[1]
        char_dim = tf.shape(c_emb)[2]
        emb_dim = c_emb.get_shape()[-1]
        c_emb = tf.reshape(c_emb, (-1, char_dim, emb_dim))

        # compute actual lengths
        tokens_lens = tf.reshape(tf.reduce_sum(tf.cast(tf.cast(ids, tf.bool), tf.int32), axis=2), (-1,))
        # to work with empty sequences if token len is zero (it means that token is padded)
        tokens_lens = tf.maximum(tf.ones_like(tokens_lens), tokens_lens)

        c_emb = variational_dropout(c_emb, keep_prob=keep_prob)

        if transform_char_emb != 0:
            # apply dense layer to transform pretrained vectors to another dim
            c_emb = transform_layer(c_emb, transform_char_emb, reuse=reuse)

        _, (state_fw, state_bw) = cudnn_bi_gru(c_emb, char_encoder_hidden_size, seq_lengths=tokens_lens,
                                               trainable_initial_states=True, reuse=reuse)
        c_emb = tf.concat([state_fw, state_bw], axis=1)
        c_emb = tf.reshape(c_emb, [-1, token_dim, 2 * char_encoder_hidden_size])

    return c_emb


def elmo_embedding_layer(tokens, elmo_module):
    # tokens are padded with empty strings
    tokens_emb = elmo_module(
        inputs={
            "tokens": tokens,
            "sequence_len": tf.reduce_sum(1 - tf.cast(tf.equal(tokens, ""), tf.int32), axis=-1)
        },
        signature="tokens",
        as_dict=True)["elmo"]

    return tokens_emb


def transform_layer(inputs, transform_hidden_size, scope='transform_layer', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        transformed = tf.layers.dense(
            tf.layers.dense(inputs, transform_hidden_size, activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                            name='transform_dense_1'),
            transform_hidden_size,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            name='transform_dense_2'
        )
    return transformed


def highway_layer(x, y, use_combinations=False, regularizer=None, scope='highway_layer', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        hidden_size = x.get_shape()[-1]
        inputs = tf.concat([x, y], axis=-1)
        if use_combinations:
            inputs = tf.concat([inputs, x * y, x - y], axis=-1)

        gate = tf.layers.dense(inputs, hidden_size, activation=tf.sigmoid,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               kernel_regularizer=regularizer,
                               )

        x_y_repr = tf.layers.dense(inputs, hidden_size, activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                   kernel_regularizer=regularizer,
                                   )

    return gate * x_y_repr + (1 - gate) * x


def pointer_net_answer_selection(q, context_repr, q_mask, c_mask, att_hidden_size, keep_prob):
    q_att = simple_attention(q, att_hidden_size, mask=q_mask, keep_prob=keep_prob)

    pointer = PtrNet(cell_size=q_att.get_shape().as_list()[-1], keep_prob=keep_prob)
    logits1, logits2 = pointer(q_att, context_repr, att_hidden_size, c_mask)

    return logits1, logits2


def mnemonic_reader_answer_selection(q, context_repr, q_mask, c_mask, att_hidden_size, keep_prob):
    init_state = simple_attention(q, att_hidden_size, mask=q_mask, keep_prob=keep_prob)
    state = tf.layers.dense(init_state, units=context_repr.get_shape().as_list()[-1],
                            kernel_regularizer=tf.nn.l2_loss)
    context_repr = variational_dropout(context_repr, keep_prob=keep_prob)
    state = tf.nn.dropout(state, keep_prob=keep_prob)
    att, logits_st = attention(context_repr, state, att_hidden_size, c_mask, use_combinations=True, scope='st_att')
    state = highway_layer(state, att, use_combinations=True, regularizer=tf.nn.l2_loss)
    state = tf.nn.dropout(state, keep_prob=keep_prob)
    _, logits_end = attention(context_repr, state, att_hidden_size, c_mask, use_combinations=True, scope='end_att')
    return logits_st, logits_end


def san_answer_selection(q, context_repr, q_mask, c_mask, n_hops, att_hidden_size, answer_cell_size, keep_prob,
                         hops_keep_prob):
    bs = tf.shape(q)[0]

    q_att = simple_attention(q, att_hidden_size, mask=q_mask, keep_prob=keep_prob)
    q_att = tf.layers.dense(q_att, units=answer_cell_size, name='init_projection')
    state = variational_dropout(q_att, keep_prob=keep_prob)
    multihop_cell = tf.nn.rnn_cell.GRUCell(num_units=answer_cell_size)

    hops_start_logits = []
    hops_end_logits = []

    for i in range(n_hops):
        x, _ = mult_attention(context_repr, state, mask=c_mask, scope='multihop_cell_att', reuse=tf.AUTO_REUSE)
        x = variational_dropout(x, keep_prob=keep_prob)
        _, state = multihop_cell(x, state)

        start_att, start_logits = mult_attention(context_repr, state, mask=c_mask, scope='start_pointer_att',
                                                 reuse=tf.AUTO_REUSE)

        _, end_logits = mult_attention(context_repr, tf.concat([state, start_att], axis=-1), mask=c_mask,
                                       scope='end_pointer_att', reuse=tf.AUTO_REUSE)

        hops_start_logits.append(start_logits)
        hops_end_logits.append(end_logits)

    hops_start_logits = tf.stack(hops_start_logits, axis=1)
    hops_end_logits = tf.stack(hops_end_logits, axis=1)

    # TODO check if dropout all zeros
    logits1 = tf.reduce_mean(tf.nn.dropout(hops_start_logits, keep_prob=hops_keep_prob, noise_shape=(bs, n_hops, 1)),
                             axis=1)

    logits2 = tf.reduce_mean(tf.nn.dropout(hops_end_logits, keep_prob=hops_keep_prob, noise_shape=(bs, n_hops, 1)),
                             axis=1)

    return logits1, logits2
