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

from logging import getLogger
from typing import List, Tuple, Optional, Union

import keras.optimizers
import numpy as np
from keras.layers import Dense, Input
from keras.layers import Activation, Concatenate, Add, Multiply, Subtract
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.regularizers import l2
from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.models.classifiers.keras_classification_model import KerasClassificationModel

log = getLogger(__name__)


@register('keras_entailment_model')
class KerasEntailmentModel(KerasClassificationModel):
    """
    Class implements Keras model for entailment task for multi-class multi-labeled pair of texts.

    Args:
        embedding_size: embedding_size from embedder in pipeline
        n_classes: number of considered classes
        model_name: particular method of this class to initialize model configuration
        optimizer: function name from keras.optimizers
        loss: function name from keras.losses.
        learning_rate: learning rate for optimizer.
        learning_rate_decay: learning rate decay for optimizer
        last_layer_activation: parameter that determines activation function after classification layer.
                For multi-label classification use `sigmoid`,
                otherwise, `softmax`.
        restore_lr: in case of loading pre-trained model \
                whether to init learning rate with the final learning rate value from saved opt
        classes: list or generator of considered classes
        text_size: maximal length of text in tokens (words),
                longer texts are cut,
                shorter ones are padded with zeros (pre-padding)
        padding: ``pre`` or ``post`` padding to use

    Attributes:
        opt: dictionary with all model parameters
        n_classes: number of considered classes
        model: keras model itself
        epochs_done: number of epochs that were done
        batches_seen: number of epochs that were seen
        train_examples_seen: number of training samples that were seen
        sess: tf session
        optimizer: keras.optimizers instance
        classes: list of considered classes
        padding: ``pre`` or ``post`` padding to use
    """

    def __init__(self,
                 **kwargs):
        """
        Initialize model using parameters
        from opt dictionary (from config), if model is being initialized from saved.
        """

        if isinstance(kwargs["in"], str):
            self.n_inputs = 1
        else:
            self.n_inputs = len(kwargs["in"])

        super().__init__(**kwargs)

    @overrides
    def train_on_batch(self, *args) -> Union[float, List[float]]:
        """
        Train the model on the given batch

        Args:

        Returns:
            metrics values on the given batch
        """
        features = [self.check_input(args[i]) for i in range(self.n_inputs)]
        labels = args[-1]

        metrics_values = self.model.train_on_batch(features, np.squeeze(np.array(labels)))
        return metrics_values

    @overrides
    def infer_on_batch(self, *args) -> \
            Union[float, List[float], np.ndarray]:
        """
        Infer the model on the given batch

        Args:

        Returns:
            metrics values on the given batch, if labels are given
            predictions, otherwise
        """
        features = [self.check_input(args[i]) for i in range(self.n_inputs)]
        if len(args) > self.n_inputs:
            labels = args[-1]
        else:
            labels = None

        if labels:
            metrics_values = self.model.test_on_batch(features, np.squeeze(np.array(labels)))
            return metrics_values
        else:
            predictions = self.model.predict(features)
            return predictions

    @overrides
    def __call__(self, *args) -> List[List[float]]:
        """
        Infer on the given data

        Args:
            *args: arguments

        Returns:
            for each sentence:
                vector of probabilities to belong with each class
                or list of labels sentence belongs with
        """
        preds = np.array(self.infer_on_batch(*args), dtype="float64").tolist()
        return preds

    @overrides
    def bigru_with_max_aver_pool_model(self, units_gru: int, dense_size: int,
                                       coef_reg_gru: float = 0., coef_reg_den: float = 0.,
                                       dropout_rate: float = 0., rec_dropout_rate: float = 0.,
                                       **kwargs) -> Model:
        """
        Method builds uncompiled model Bidirectional GRU with concatenation of max and average pooling after BiGRU.

        Args:
            units_gru: number of units for GRU.
            dense_size: number of units for dense layer.
            coef_reg_gru: l2-regularization coefficient for GRU. Default: ``0.0``.
            coef_reg_den: l2-regularization coefficient for dense layers. Default: ``0.0``.
            dropout_rate: dropout rate to be used after BiGRU and between dense layers. Default: ``0.0``.
            rec_dropout_rate: dropout rate for GRU. Default: ``0.0``.
            kwargs: other non-used parameters

        Returns:
            keras.models.Model: uncompiled instance of Keras Model
        """

        inputs = [Input(shape=(self.opt['text_size'], self.opt['embedding_size']))
                  for _ in range(self.n_inputs)]
        outputs = [inputs[i] for i in range(self.n_inputs)]

        dropout = Dropout(rate=dropout_rate)
        bigru = Bidirectional(GRU(units_gru, activation='tanh',
                                  return_sequences=True,
                                  return_state=True,
                                  kernel_regularizer=l2(coef_reg_gru),
                                  dropout=dropout_rate,
                                  recurrent_dropout=rec_dropout_rate))
        maxpool = GlobalMaxPooling1D()
        averpool = GlobalAveragePooling1D()
        concat = Concatenate(axis=-1)

        full_outputs = []

        for j in range(self.n_inputs):
            output = dropout(outputs[j])

            output, state1, state2 = bigru(output)

            output1 = maxpool(output)
            output2 = averpool(output)

            output = concat([output1, output2, state1, state2])
            full_outputs.append(output)

        output = Concatenate()(full_outputs)

        output = Dropout(rate=dropout_rate)(output)
        output = Dense(dense_size, activation=None,
                       kernel_regularizer=l2(coef_reg_den))(output)
        output = Activation('relu')(output)
        output = Dropout(rate=dropout_rate)(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(coef_reg_den))(output)
        act_output = Activation(self.opt.get("last_layer_activation", "sigmoid"))(output)
        model = Model(inputs=inputs, outputs=act_output)
        return model
