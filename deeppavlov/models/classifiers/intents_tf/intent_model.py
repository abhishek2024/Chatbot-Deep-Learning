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

import numpy as np

from deeppavlov.core.common.attributes import check_attr_true
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.tf_model import TFModel
from deeppavlov.models.classifiers.intents.utils import labels2onehot, log_metrics, proba2labels
from deeppavlov.models.embedders.fasttext_embedder import FasttextEmbedder
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
                 train_now=False,
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
        self.opt = {"confident_threshold": 0.5,
                    "optimizer": "Adam",
                    "lear_rate": 0.1,
                    "lear_rate_decay": 0.1,
                    "coef_reg_cnn": 1e-4,
                    "coef_reg_den": 1e-4,
                    "dropout_rate": 0.5}
        self.opt.update(params)

    def run_sess(self):
        pass

    def _build_graph(self):

        self._add_placeholders()

        units = []
        for i, kern_size_i in enumerate(params['kernel_sizes_cnn']):
            with tf.variable_scope('ConvNet_{}'.format(i)):
                units_i = \
                    dense_convolutional_network(self.X_ph,
                                                n_filters=params['filters_cnn'],
                                                n_layers=1,
                                                filter_width=kern_size_i,
                                                use_batch_norm=True,
                                                training=self.train_mode_ph)
                units_i = tf.keras.layers.GlobalMaxPool1D()(units_i)
                units.append(units_i)
            
        units = tf.concat(units, 1) 
        with tf.variable_scope('Classifier'):
            with tf.variable_scope('Layer_1'):
                units = tf.nn.dropout(units, self.dropout_ph)
                units = tf.layers.dense(units,
                                        self.opt['dense_size'],
                                        kernel_initializer=xavier_initializer())  
                units = tf.layers.batch_normalization(units,
                                                      training=self.train_mode_ph)
                units = tf.nn.relu(units)
            with tf.variable_scope('Layer_2'):
                units = tf.nn.dropout(units, self.dropout_ph)
                units = tf.layers.dense(units,
                                        self.opt['num_classes'],
                                        kernel_initializer=xavier_initializer())  
                units = tf.layers.batch_normalization(units,
                                                      training=self.train_mode_ph)

        self.probs_op = tf.nn.sigmoid(units)
        self.prediction_op = tf.argmax(prediction, axis=-1, name='prediction')
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=units,
                                                                labels=self.y_ph)
        self.loss_op = tf.reduce_mean(cross_entropy, name='loss')
        self.train_op = self._get_train_op(self.loss_op, self.learning_rate_ph)

    def _add_placeholders(self):
        self.X_ph = \
            tf.placeholder(tf.float32, 
                           [None, None, self.opt['embedding_size']],
                           name='features')
        self.train_mode_ph = tf.placeholder_with_default(False, shape=[])
        self.dropout_ph = tf.placeholder_with_default(1.0, shape=[])
        self.learning_rate_ph = \
            tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
        self.y_ph = \
            tf.placeholder(tf.float32, [None, self.opt['num_classes']])

    def _get_train_op(self, loss, learning_rate, optimizer=None, clip_norm=1.):
        """Get train operation for given loss

        Args:
            loss: loss function, tf tensor of scalar
            learning_rate: scalar or placeholder
            optimizer: instance of tf.train.Optimizer, Adam, by default

        Returns:
            train_op
        """

        optimizer = optimizer or tf.train.AdamOptimizer
        optimizer = optimizer(learning_rate)
        if clip_norm is not None:
            grads_and_vars = \
                optimizer.compute_gradients(loss, tf.trainable_variables())
            grads_and_vars = [(tf.clip_by_norm(grad, clip_norm), var)\
                              for grad, var in grads_and_vars]
            return optimizer.apply_gradients(grads_and_vars, name='train_op')
        return optimizer.minimize(loss)

    def texts2vec(self, utterances):
        """
        Convert texts to vector representations using embedder and
        padding up to max(max_i(len(tokens_i)), 2) tokens

        Args:
            sentences: list of texts

        Returns:
            array of embedded texts
        """
        batch_size = len(sentences)
        max_utt_len = max([len(utt) for utt in utterances] + [2])

        x = np.zeros([batch_size, max_utt_len, self.opt['embedding_size']],
                     dtype=np.float32)
        mask = np.zeros([batch_size, max_utt_len, self.opt['embedding_size']],
                        dtype=np.int32)
        
        for i, utt in enumerate(utterances):
            x[i, :len(utt)] = self.fasttext_model.infer(utt)
            mask[i, :len(utt)] = 1

        return x, mask

    @check_attr_true('train_now')
    def train_on_batch(self, batch):
        batch_x, batch_y = batch
        x, mask = self.texts2vec(self.tokenizer.infer(batch_x))
        y = labels2onehot(batch_y, classes=self.classes)

        feed_dict = self._get_feed_dict(x,
                                        y=y,
                                        learning_rate=self._get_learning_rate(),
                                        dropout_rate=self.opt['dropout_rate'],
                                        train_mode=True)
        loss, _ = self.sess.run([self.loss_op, self.train_op],
                                feed_dict=feed_dict)
        return loss

    def _get_learning_rate(self):
        #TODO: decaying learning rate
        return self.opt['lear_rate']

    def _get_feed_dict(self, x, y=None, learning_rate=None, dropout_rate=None,
                       train_mode=False):
        feed_dict = {
            self.X_ph = x,
            self.train_mode_ph = train_mode
        }
        if y is not None:
            feed_dict[self.y_ph] = y
        if learning_rate is not None:
            feed_dict[self.learning_rate_ph] = learning_rate
        if self.train_mode and dropout_rate is not None:
            feed_dict[self.dropout_ph] = dropout_rate
        else:
            feed_dict[self.dropout_ph] = 1.
        return feed_dict

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
