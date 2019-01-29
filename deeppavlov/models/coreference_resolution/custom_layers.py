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

import numpy as np
import tensorflow as tf


seed = 5
np.random.seed(seed)


def flatten(l):
    """Expands list"""
    return [item for sublist in l for item in sublist]


def normalize(v):
    """Normalize input tensor"""
    norm = np.linalg.norm(v)
    if norm > 0:
        return v / norm
    else:
        return v


def maybe_divide(x, y):
    return 0 if y == 0 else x / float(y)


def projection(inputs, output_size):
    """ Returns outputs of fully-connected network """
    return ffnn(inputs, 0, -1, output_size, dropout=None)


def shape(x, dim):
    """ Returns shape of tensor """
    return x.get_shape()[dim].value or tf.shape(x)[dim]


#  Networks
def ffnn(inputs, num_hidden_layers, hidden_size, output_size, dropout):
    """
    Creates fully_connected network graph with needed parameters.
    Args:
        inputs: shape of input tensor, rank of input >3 not supported
        num_hidden_layers: int32
        hidden_size: int32
        output_size: int32
        dropout: int32, dropout value

    Returns: network output, [output_size]

    """

    # inputs = tf.cast(inputs, tf.float32)
    if len(inputs.get_shape()) > 2:
        current_inputs = tf.reshape(inputs, [-1, shape(inputs, -1)])
    else:
        current_inputs = inputs

    for i in range(num_hidden_layers):
        hidden_weights = tf.get_variable("hidden_weights_{}".format(i), [shape(current_inputs, 1), hidden_size],
                                         dtype=tf.float64)
        hidden_bias = tf.get_variable("hidden_bias_{}".format(i), [hidden_size], dtype=tf.float64)

        current_outputs = tf.nn.relu(tf.matmul(current_inputs, hidden_weights) + hidden_bias)

        if dropout is not None:
            current_outputs = tf.nn.dropout(current_outputs, dropout)
        current_inputs = current_outputs

    output_weights = tf.get_variable("output_weights", [shape(current_inputs, 1), output_size],
                                     dtype=tf.float64)
    output_bias = tf.get_variable("output_bias", [output_size], dtype=tf.float64)
    outputs = tf.matmul(current_inputs, output_weights) + output_bias
    # outputs = tf.cast(outputs, tf.float64)
    if len(inputs.get_shape()) == 3:
        outputs = tf.reshape(outputs, [shape(inputs, 0), shape(inputs, 1), output_size])
    elif len(inputs.get_shape()) > 3:
        raise ValueError("FFNN with rank {} not supported".format(len(inputs.get_shape())))
    return outputs


def cnn(inputs, filter_sizes, num_filters):
    """
    Creates convolutional network graph with needed parameters.
    Args:
        inputs: shape of input tensor
        filter_sizes: list of shapes of filters
        num_filters: amount of filters

    Returns: network output, [num_words, num_filters * len(filter_sizes)]

    """
    input_size = shape(inputs, 2)
    outputs = []

    # TODO: del the tf.cast(float32) when https://github.com/tensorflow/tensorflow/pull/12943 will be done
    for i, filter_size in enumerate(filter_sizes):
        with tf.variable_scope("conv_{}".format(i)):
            w = tf.get_variable("w", [filter_size, input_size, num_filters])
            b = tf.get_variable("b", [num_filters], dtype=tf.float64)
        conv = tf.nn.conv1d(tf.cast(inputs, tf.float32), w, stride=1, padding="VALID")
        # [num_words, num_chars - filter_size, num_filters]
        h = tf.nn.relu(tf.nn.bias_add(tf.cast(conv, tf.float64), b))
        # [num_words, num_chars - filter_size, num_filters]
        pooled = tf.reduce_max(h, 1)  # [num_words, num_filters]
        outputs.append(pooled)
    return tf.concat(outputs, 1)  # [num_words, num_filters * len(filter_sizes)]


class CustomLSTMCell(tf.contrib.rnn.RNNCell):
    """Bi-LSTM"""

    def __init__(self, num_units, batch_size, dropout):
        """Initialize graph"""
        self._num_units = num_units
        self._dropout = dropout
        self._dropout_mask = tf.nn.dropout(tf.ones([batch_size, self.output_size], dtype=tf.float64), dropout)
        self._initializer = self._block_orthonormal_initializer([self.output_size] * 3)
        initial_cell_state = tf.get_variable("lstm_initial_cell_state", [1, self.output_size], dtype=tf.float64)
        initial_hidden_state = tf.get_variable("lstm_initial_hidden_state", [1, self.output_size], dtype=tf.float64)
        self._initial_state = tf.contrib.rnn.LSTMStateTuple(initial_cell_state, initial_hidden_state)

    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self.output_size, self.output_size)

    @property
    def output_size(self):
        return self._num_units

    @property
    def initial_state(self):
        return self._initial_state

    def preprocess_input(self, inputs):
        return projection(inputs, 3 * self.output_size)

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # "CustomLSTMCell"
            c, h = state
            h *= self._dropout_mask
            projected_h = projection(h, 3 * self.output_size)
            concat = inputs + projected_h
            i, j, o = tf.split(concat, num_or_size_splits=3, axis=1)
            i = tf.sigmoid(i)
            new_c = (1 - i) * c + i * tf.tanh(j)
            new_h = tf.tanh(new_c) * tf.sigmoid(o)
            new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
            return new_h, new_state

    @staticmethod
    def _orthonormal_initializer(scale=1.0):
        def _initializer(shape_):
            M1 = np.random.randn(shape_[0], shape_[0]).astype(np.float64)
            M2 = np.random.randn(shape_[1], shape_[1]).astype(np.float64)
            Q1, R1 = np.linalg.qr(M1)
            Q2, R2 = np.linalg.qr(M2)
            Q1 = Q1 * np.sign(np.diag(R1))
            Q2 = Q2 * np.sign(np.diag(R2))
            n_min = min(shape_[0], shape_[1])
            params = np.dot(Q1[:, :n_min], Q2[:n_min, :]) * scale
            return params

        return _initializer

    def _block_orthonormal_initializer(self, output_sizes):
        def _initializer(_shape):
            assert len(_shape) == 2
            assert sum(output_sizes) == _shape[1]
            initializer = self._orthonormal_initializer()
            params = np.concatenate([initializer([_shape[0], o]) for o in output_sizes], 1)
            return params

        return _initializer
