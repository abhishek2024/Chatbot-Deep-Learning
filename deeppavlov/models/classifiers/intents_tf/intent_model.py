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
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

from deeppavlov.core.common.attributes import check_attr_true
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.tf_model import TFModel
from deeppavlov.models.embedders.fasttext_embedder import FasttextEmbedder
from deeppavlov.models.tokenizers.nltk_tokenizer import NLTKTokenizer
from deeppavlov.models.classifiers.intents_tf.utils import labels2onehot, probs2labels
from deeppavlov.models.ner.layers import dense_convolutional_network

from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


@register('tf_intent_model')
class TFIntentModel(TFModel):

    def __init__(self,
                 vocabs,
                 embedder: FasttextEmbedder,
                 tokenizer: NLTKTokenizer,
                 save_path,
                 load_path=None,
                 train_now=False,
                 gradient_flow=False,
                 **params):

        super().__init__(save_path=save_path,
                         load_path=load_path,
                         train_now=train_now,
                         mode=params['mode'])

        # Initialize parameters
        self._init_params(params)

        # Tokenizer and vocabulary of classes
        self.tokenizer = tokenizer
        self.classes = sorted(list(vocabs["classes_vocab"].keys()))
        self.opt['num_classes'] = len(self.classes)
        self.fasttext_model = embedder
        self.opt['embedding_size'] = self.fasttext_model.dim

        # build computational graph
        if not gradient_flow:
            self._build_graph()
            self.init_session()

    def init_session(self, session=None):
        self.sess = session or tf.Session()
        class_name = self.__class__.__name__
        if self.get_checkpoint_state():
            log.info("[initializing `{}` from saved]".format(class_name))
            self.load()
        else:
            log.info("[initializing `{}` from scratch]".format(class_name))
            self.sess.run(tf.variables_initializer(
                tf.global_variables(self.scope_name)))

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

        logits = self._build_body()

        self.probs_op = tf.nn.sigmoid(logits)

        self.loss_op = self._get_loss_op(logits)

        optimizer = None
        if self.opt['optimizer'] == 'Adam':
            optimizer = tf.train.AdamOptimizer
        self.train_op = self._get_train_op(self.loss_op,
                                           learning_rate=self.learning_rate_ph,
                                           optimizer=optimizer)
        return logits

    def _get_loss_op(self, logits):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                labels=self.y_ph)
        return tf.reduce_mean(cross_entropy, name='loss')

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

    def _build_body(self):
        units = []
        l2 = tf.contrib.layers.l2_regularizer(self.opt['coef_reg_cnn'])
        for i, kern_size_i in enumerate(self.opt['kernel_sizes_cnn']):
            with tf.variable_scope('ConvNet_{}'.format(i)):
                units_i = tf.layers.conv1d(self.X_ph,
                                           filters=self.opt['filters_cnn'],
                                           kernel_size=kern_size_i,
                                           padding='same',
                                           kernel_regularizer=l2,
                                           kernel_initializer=xavier_initializer())
                units_i = tf.layers.batch_normalization(units_i,
                                                        training=self.train_mode_ph)
                units_i = tf.nn.relu(units_i)
                units_i = tf.reduce_max(units_i, axis=1)
                units.append(units_i)
            
        units = tf.concat(units, 1) 
        with tf.variable_scope('Classifier'):
            with tf.variable_scope('Layer_1'):
                units = tf.nn.dropout(units, self.dropout_ph)
                l2 = tf.contrib.layers.l2_regularizer(self.opt['coef_reg_den'])
                units = tf.layers.dense(units,
                                        self.opt['dense_size'],
                                        kernel_regularizer=l2,
                                        kernel_initializer=xavier_initializer())  
                units = tf.layers.batch_normalization(units,
                                                      training=self.train_mode_ph)
                units = tf.nn.relu(units)
            with tf.variable_scope('Layer_2'):
                units = tf.nn.dropout(units, self.dropout_ph)
                l2 = tf.contrib.layers.l2_regularizer(self.opt['coef_reg_den'])
                units = tf.layers.dense(units,
                                        self.opt['num_classes'],
                                        kernel_regularizer=l2,
                                        kernel_initializer=xavier_initializer())  
                units = tf.layers.batch_normalization(units,
                                                      training=self.train_mode_ph)
        return units

    def _get_train_op(self, loss, learning_rate, optimizer=None, scope_names=None,
                      clip_norm=1.):
        """Construct training operation that is using `scope_names` scopes with
        - gradient clipping and
        - batch normalization fix.

        Args:
            loss: loss function, tf tensor of scalar
            learning_rate: scalar or placeholder
            scope_names: list of trainable scope names
            optimizer: instance of tf.train.Optimizer

        Returns:
            train_op
        """
        scope_names = [self.scope_name]
        vars = self._get_trainable_variables(scope_names) 

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                             scope=self.scope_name)
        with tf.control_dependencies(extra_update_ops):
            with tf.variable_scope('Optimizer'):

                optimizer = optimizer(learning_rate)

                if clip_norm is not None:
                    grads_and_vars = optimizer.compute_gradients(loss, vars)
                    grads_and_vars = [(tf.clip_by_norm(grad, clip_norm), var)\
                                      for grad, var in grads_and_vars]
                    return optimizer.apply_gradients(grads_and_vars,
                                                     name='train_op')

                return optimizer.minimize(loss, var_list=vars)

    @staticmethod
    def _get_trainable_variables(scope_names=None):
        all_vars = tf.trainable_variables()
        if scope_names is not None:
            vars_to_train = set()
            for sn in scope_names:
                scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                               scope=sn)
                vars_to_train.update(scope_vars)
            return list(vars_to_train)
        else:
            return all_vars

    def build_batch(self, x, y=None):
        batch_x, _ = self.texts2vec(self.tokenizer.infer(x))
        if y is not None:
            batch_y = labels2onehot(y, classes=self.classes)

            feed_dict = self._get_feed_dict(batch_x,
                                            y=batch_y,
                                            learning_rate=self._get_learning_rate(),
                                            dropout_rate=self.opt['dropout_rate'],
                                            train_mode=True)
            fetches = [self.loss_op, self.train_op]
        else:
            feed_dict = self._get_feed_dict(batch_x, train_mode=False)
            fetches = []
        return fetches, feed_dict

    @check_attr_true('train_now')
    def train_on_batch(self, batch):
        batch_x, batch_y = batch
        x, _ = self.texts2vec(self.tokenizer.infer(batch_x))
        y = labels2onehot(batch_y, classes=self.classes)

        feed_dict = self._get_feed_dict(x,
                                        y=y,
                                        learning_rate=self._get_learning_rate(),
                                        dropout_rate=self.opt['dropout_rate'],
                                        train_mode=True)
        loss, _ = self.sess.run([self.loss_op, self.train_op],
                                feed_dict=feed_dict)
        return loss

    def texts2vec(self, utterances):
        """
        Convert texts to vector representations using embedder and
        padding up to max(max_i(len(tokens_i)), 2) tokens

        Args:
            sentences: list of texts

        Returns:
            array of embedded texts
        """
        batch_size = len(utterances)
        max_utt_len = max([len(utt.split()) for utt in utterances] + [2])

        x = np.zeros([batch_size, max_utt_len, self.opt['embedding_size']],
                     dtype=np.float32)
        mask = np.zeros([batch_size, max_utt_len, self.opt['embedding_size']],
                        dtype=np.int32)
        
        for i, utt in enumerate(utterances):
            if utt:
                x_utt = self.fasttext_model.infer(utt)
                x[i, :len(x_utt)] = x_utt
                mask[i, :len(utt)] = 1

        return x, mask

    def _get_learning_rate(self):
        #TODO: decaying learning rate
        return self.opt['lear_rate']

    def _get_feed_dict(self, x, y=None, learning_rate=None, dropout_rate=None,
                       train_mode=False):
        feed_dict = {
            self.X_ph: x,
            self.train_mode_ph: train_mode
        }
        if y is not None:
            feed_dict[self.y_ph] = y
        if learning_rate is not None:
            feed_dict[self.learning_rate_ph] = learning_rate
        if train_mode and dropout_rate is not None:
            feed_dict[self.dropout_ph] = dropout_rate
        else:
            feed_dict[self.dropout_ph] = 1.
        return feed_dict

    def infer(self, data, *args, **kwargs):
        if isinstance(data, str):
            return self.infer_on_batch([data], *args, **kwargs)[0]
        return self.infer_on_batch(data, *args, **kwargs)

    def infer_on_batch(self, batch, prob=False):
        x, _ = self.texts2vec(self.tokenizer.infer(batch))

        feed_dict = self._get_feed_dict(x, train_mode=False)
        preds = self.sess.run(self.probs_op, feed_dict=feed_dict) 

        if not prob:
            return probs2labels(preds,
                                threshold=self.opt['confident_threshold'],
                                classes=self.classes)
        return preds

    def _forward(self, x):
        pass

    def _train_step(self, x, y):
        pass

    def reset(self):
        pass

    def shutdown(self):
        self.sess.close()
