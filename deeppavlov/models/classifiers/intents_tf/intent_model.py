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

import sys
import inspect

from typing import Dict
import numpy as np
from keras.layers import Dense, Input, concatenate, Activation
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalMaxPooling1D, MaxPooling1D
from keras.models import Model
from keras.regularizers import l2

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.attributes import check_attr_true
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.tf_model import TFModel
from deeppavlov.models.classifiers.intents.utils import labels2onehot, log_metrics, proba2labels
from deeppavlov.models.embedders.fasttext_embedder import FasttextEmbedder
from deeppavlov.models.classifiers.intents.utils import md5_hashsum
from deeppavlov.models.tokenizers.nltk_tokenizer import NLTKTokenizer
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


@register('intent_model_tf')
class IntentModel(TFModel):

    def __init__(self,
                 vocabs,
                 embedder: FasttextEmbedder,
                 tokenizer: NLTKTokenizer,
                 save_path,
                 load_path=None,
                 train_now=False
                 **params):

        super().__init__(save_path=save_path,
                         load_path=load_path,
                         train_now=train_now,
                         mode=params['mode'])

        # Initialize parameters
        self._init_params(params)

        # Tokenizer and vocabulary of classes
        self.tokenizer = tokenizer
        self.classes = np.sort(list(vocabs["classes_vocab"].keys()))
        self.opt['num_classes'] = self.classes.shape[0]
        self.fasttext_model = embedder
        self.opt['embedding_size'] = self.fasttext_model.dim

        # build computational graph
        self._build_graph()
        # initialize session
        self.sess = tf.Session()
        
        class_name = self.__class__.__name__
        if self.get_checkpoint_state():
            log.info("[initializing `{}` from saved]".format(class_name))
            self.load()
        else:
            log.info("[initializing `{}` from scratch]".format(class_name))
            self.sess.run(tf.global_variables_initializer())

    def _init_params(self, params={}):
        self.opt = {"lear_metrics": ["binary_accuracy"],
                    "confident_threshold": 0.5,
                    "optimizer": "Adam",
                    "lear_rate": 0.1,
                    "lear_rate_decay": 0.1,
                    "loss": "binary_crossentropy",
                    "coef_reg_cnn": 1e-4,
                    "coef_reg_den": 1e-4,
                    "dropout_rate": 0.5}
        self.opt.update(params)

    def run_sess(self):
        pass

    def _build_graph(self):

        self._add_placeholders()

    def _add_placeholders(self):
        return

    def texts2vec(self, sentences):
        """
        Convert texts to vector representations using embedder and
        padding up to self.opt["text_size"] tokens

        Args:
            sentences: list of texts

        Returns:
            array of embedded texts
        """
        embeddings_batch = []
        for sen in sentences:
            tokens = [el for el in sen.split() if el]
            if len(tokens) > self.opt['text_size']:
                tokens = tokens[:self.opt['text_size']]

            embeddings = self.fasttext_model.infer(' '.join(tokens))
            if len(tokens) < self.opt['text_size']:
                pads = [np.zeros(self.opt['embedding_size'])
                        for _ in range(self.opt['text_size'] - len(tokens))]
                embeddings = pads + embeddings

            embeddings = np.asarray(embeddings)
            embeddings_batch.append(embeddings)

        embeddings_batch = np.asarray(embeddings_batch)
        return embeddings_batch

    @check_attr_true('train_now')
    def train_on_batch(self, batch):
        x, y = batch
        features = self.texts2vec(self.tokenizer.infer(x))
        onehot_labels = labels2onehot(y, classes=self.classes)
        self._train_step(features, onehot_labels)

    def _train_step(self, features, labels):
        return

    def infer(self, data, *args, **kwargs):
        if isinstance(data, str):
            return self.infer_on_batch([data], *args, **kwargs)[0]
        return self.infer_on_batch(data, *args, **kwargs)

    def infer_on_batch(self, batch, prob=False):
        features = self.texts2vec(self.tokenizer.infer(batch))
        preds = self._forward(features)
        if prob:
            return preds
        return proba2labels(preds,
                            confident_threshold=self.opt['confident_threshold'],
                            classes=self.classes)

    def _forward(self, features):
        return

    def cnn_model(self, params):
        """
        Build un-compiled model of shallow-and-wide CNN
        Args:
            params: dictionary of parameters for NN

        Returns:
            Un-compiled model
        """

        inp = Input(shape=(params['text_size'], params['embedding_size']))

        outputs = []
        for i in range(len(params['kernel_sizes_cnn'])):
            output_i = Conv1D(params['filters_cnn'], kernel_size=params['kernel_sizes_cnn'][i],
                              activation=None,
                              kernel_regularizer=l2(params['coef_reg_cnn']),
                              padding='same')(inp)
            output_i = BatchNormalization()(output_i)
            output_i = Activation('relu')(output_i)
            output_i = GlobalMaxPooling1D()(output_i)
            outputs.append(output_i)

        output = concatenate(outputs, axis=1)

        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(params['dense_size'], activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        output = BatchNormalization()(output)
        act_output = Activation('sigmoid')(output)
        model = Model(inputs=inp, outputs=act_output)
        return model

    def dcnn_model(self, params):
        """
        Build un-compiled model of deep CNN
        Args:
            params: dictionary of parameters for NN

        Returns:
            Un-compiled model
        """

        if type(self.opt['filters_cnn']) is str:
            self.opt['filters_cnn'] = list(map(int, self.opt['filters_cnn'].split()))

        inp = Input(shape=(params['text_size'], params['embedding_size']))

        output = inp

        for i in range(len(params['kernel_sizes_cnn'])):
            output = Conv1D(params['filters_cnn'][i], kernel_size=params['kernel_sizes_cnn'][i],
                            activation=None,
                            kernel_regularizer=l2(params['coef_reg_cnn']),
                            padding='same')(output)
            output = BatchNormalization()(output)
            output = Activation('relu')(output)
            output = MaxPooling1D()(output)

        output = GlobalMaxPooling1D()(output)
        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(params['dense_size'], activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        output = BatchNormalization()(output)
        act_output = Activation('sigmoid')(output)
        model = Model(inputs=inp, outputs=act_output)
        return model

    def shutdown(self):
        self.sess.close()
