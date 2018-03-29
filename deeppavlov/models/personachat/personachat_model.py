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

from pathlib import Path
from copy import deepcopy
import random
import uuid
import tensorflow as tf
import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.models.tf_model import TFModel
from deeppavlov.models.personachat.utils import CudnnGRU, CudnnLSTM, dot_attention, simple_attention, attention


@register('personachat_model')
class PersonaChatModel(TFModel):
    def __init__(self, **kwargs):
        self.opt = deepcopy(kwargs)
        self.init_word_emb = self.opt['word_emb']
        self.init_char_emb = self.opt['char_emb']
        self.seq_len_limit = self.opt['seq_len_limit']
        self.char_limit = self.opt['char_limit']
        self.char_hidden_size = self.opt['char_hidden_size']
        self.hidden_size = self.opt['encoder_hidden_size']
        self.attention_hidden_size = self.opt['attention_hidden_size']
        self.keep_prob = self.opt['keep_prob']
        self.learning_rate = self.opt['learning_rate']
        self.grad_clip = self.opt['grad_clip']
        self.teacher_forcing = self.opt['teacher_forcing']
        self.teacher_forcing_rate = self.opt['teacher_forcing_rate']
        self.use_match_layer = self.opt['use_match_layer']
        self.cell_type = self.opt['cell_type']
        self.vocab_size, self.word_emb_dim = self.init_word_emb.shape
        self.char_emb_dim = self.init_char_emb.shape[1]

        self.sess_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.sess_config)

        if self.cell_type == "LSTM":
            self.cudnn_cell = CudnnLSTM
            self.tf_cell = tf.nn.rnn_cell.LSTMCell
        elif self.cell_type == "GRU":
            self.cudnn_cell = CudnnGRU
            self.tf_cell = tf.nn.rnn_cell.GRUCell
        else:
            raise RuntimeError("Unknown cell type: {}".format(self.cell_type))

        self._init_graph()

        self.train_op = self.get_train_op(self.loss, self.lr_ph,
                                          optimizer=tf.train.AdamOptimizer,
                                          clip_norm=self.grad_clip)

        self.sess.run(tf.global_variables_initializer())

        self.summary_writer = None

        self.step = 0

        super().__init__(**kwargs)

        self.run_id = str(uuid.uuid4())
        self.log_path = str(expand_path(Path(self.opt['save_path']) / '{}'.format(self.run_id)))

        if kwargs['mode'] == 'train':
            if self.load_path is None:
                self.save_path = expand_path(Path(self.opt['save_path']) / '{}_model'.format(self.run_id) / 'model')
            else:
                self.save_path = self.load_path
        # Try to load the model (if there are some model files the model will be loaded from them)
        if self.load_path is not None:
            self.load()

    def _init_graph(self):
        self._init_placeholders()

        self.word_emb = tf.get_variable("word_emb", initializer=tf.constant(self.init_word_emb, dtype=tf.float32),
                                        trainable=False)
        #self.char_emb = tf.get_variable("char_emb", initializer=tf.constant(self.init_char_emb, dtype=tf.float32),
        #                                trainable=self.opt['train_char_emb'])

        self.utt_mask = tf.cast(self.utt_ph, tf.bool)
        self.persona_mask = tf.cast(self.persona_ph, tf.bool)
        self.utt_len = tf.reduce_sum(tf.cast(self.utt_mask, tf.int32), axis=1)
        self.persona_len = tf.reduce_sum(tf.cast(self.persona_mask, tf.int32), axis=1)

        bs = tf.shape(self.utt_ph)[0]
        self.utt_maxlen = tf.reduce_max(self.utt_len)
        self.persona_maxlen = tf.reduce_max(self.persona_len)
        self.utt = tf.slice(self.utt_ph, [0, 0], [bs, self.utt_maxlen])
        self.persona = tf.slice(self.persona_ph, [0, 0], [bs, self.persona_maxlen])
        self.utt_mask = tf.slice(self.utt_mask, [0, 0], [bs, self.utt_maxlen])
        self.persona_mask = tf.slice(self.persona_mask, [0, 0], [bs, self.persona_maxlen])

        self.utt_c = tf.slice(self.utt_c_ph, [0, 0, 0], [bs, self.utt_maxlen, self.char_limit])
        self.persona_c = tf.slice(self.persona_c_ph, [0, 0, 0], [bs, self.persona_maxlen, self.char_limit])
        self.utt_c_len = tf.reshape(tf.reduce_sum(tf.cast(tf.cast(self.utt_c, tf.bool), tf.int32), axis=2), [-1])
        self.persona_c_len = tf.reshape(tf.reduce_sum(tf.cast(tf.cast(self.persona_c, tf.bool), tf.int32), axis=2), [-1])

        self.y_mask = tf.cast(tf.cast(self.y_ph, tf.bool), tf.float32)
        self.y_len = tf.reduce_sum(tf.cast(self.y_mask, tf.int32), axis=1)
        # to add loss on last <PAD> symbol, to predict eos
        self.y_mask += tf.one_hot(self.y_len, depth=self.seq_len_limit, dtype=tf.float32)
        self.y = tf.slice(self.y_ph, [0, 0], [bs, self.seq_len_limit])

        with tf.variable_scope("emb"):
            """
            with tf.variable_scope("char"):
                persona_c_emb = tf.reshape(tf.nn.embedding_lookup(self.char_emb, self.persona_c),
                                    [bs * self.persona_maxlen, self.char_limit, self.char_emb_dim])
                utt_c_emb = tf.reshape(tf.nn.embedding_lookup(self.char_emb, self.utt_c),
                                    [bs * self.utt_maxlen, self.char_limit, self.char_emb_dim])
                persona_c_emb = tf.nn.dropout(persona_c_emb, keep_prob=self.keep_prob_ph,
                                       noise_shape=[bs * self.persona_maxlen, 1, self.char_emb_dim])
                utt_c_emb = tf.nn.dropout(utt_c_emb, keep_prob=self.keep_prob_ph,
                                       noise_shape=[bs * self.utt_maxlen, 1, self.char_emb_dim])

                cell_fw = tf.contrib.rnn.GRUCell(self.char_hidden_size)
                cell_bw = tf.contrib.rnn.GRUCell(self.char_hidden_size)
                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, persona_c_emb, self.persona_c_len, dtype=tf.float32)
                persona_c_emb = tf.concat([state_fw, state_bw], axis=1)
                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, utt_c_emb, self.utt_c_len, dtype=tf.float32)
                utt_c_emb = tf.concat([state_fw, state_bw], axis=1)
                persona_c_emb = tf.reshape(persona_c_emb, [bs, self.persona_maxlen, 2 * self.char_hidden_size])
                utt_c_emb = tf.reshape(utt_c_emb, [bs, self.utt_maxlen, 2 * self.char_hidden_size])
            """
            with tf.name_scope("word"):
                persona_emb = tf.nn.embedding_lookup(self.word_emb, self.persona)
                utt_emb = tf.nn.embedding_lookup(self.word_emb, self.utt)

            #persona_emb = tf.concat([persona_emb, persona_c_emb], axis=2)
            #utt_emb = tf.concat([utt_emb, utt_c_emb], axis=2)

        with tf.variable_scope("encoding"):
            rnn = self.cudnn_cell(num_layers=2, num_units=self.hidden_size, batch_size=bs,
                           input_size=persona_emb.get_shape().as_list()[-1],
                           keep_prob=self.keep_prob_ph)
            persona = rnn(persona_emb, seq_len=self.persona_len)
            utt = rnn(utt_emb, seq_len=self.utt_len)
        
        with tf.variable_scope("attention"):
            pu_att = dot_attention(persona, utt, mask=self.utt_mask, att_size=self.attention_hidden_size,
                                   keep_prob=self.keep_prob_ph)
            rnn = self.cudnn_cell(num_layers=1, num_units=self.hidden_size, batch_size=bs,
                           input_size=pu_att.get_shape().as_list()[-1], keep_prob=self.keep_prob_ph)
            att = rnn(pu_att, seq_len=self.persona_len)

        if self.use_match_layer:
            with tf.variable_scope("match"):
                self_att = dot_attention(att, att, mask=self.persona_mask, att_size=self.attention_hidden_size,
                                         keep_prob=self.keep_prob_ph)
                rnn = self.cudnn_cell(num_layers=1, num_units=self.hidden_size, batch_size=bs,
                               input_size=self_att.get_shape().as_list()[-1], keep_prob=self.keep_prob_ph)
                match = rnn(self_att, seq_len=self.persona_len)

        with tf.variable_scope('decoder'):
            cell_state = simple_attention(utt, self.hidden_size, mask=self.utt_mask, keep_prob=self.keep_prob_ph)
            decoder_cell_size = self.hidden_size
            cell_state = tf.layers.dense(cell_state, decoder_cell_size, kernel_initializer=tf.contrib.layers.xavier_initializer(),)
            if self.cell_type == 'GRU':
                cell_state = (cell_state, cell_state)
                cell_out = cell_state[0]
            elif self.cell_type == 'LSTM':
                cell_state = ((cell_state, cell_state), (cell_state, cell_state))
                cell_out = cell_state[0][0]
            else:
                raise RuntimeError('Unknown cell type: {}'.format(self.cell_type))

            decoder_cell = tf.contrib.rnn.MultiRNNCell([
                self.tf_cell(decoder_cell_size),
                self.tf_cell(decoder_cell_size)])

            if self.use_match_layer:
                persona_do = tf.nn.dropout(match, keep_prob=self.keep_prob, noise_shape=[bs, 1, tf.shape(match)[-1]])
            else:
                persona_do = tf.nn.dropout(att, keep_prob=self.keep_prob, noise_shape=[bs, 1, tf.shape(att)[-1]])

            utt_do = tf.nn.dropout(utt, keep_prob=self.keep_prob, noise_shape=[bs, 1, tf.shape(utt)[-1]])
            dropout_mask = tf.nn.dropout(tf.ones([bs, tf.shape(cell_state[-1])[-1]], dtype=tf.float32), keep_prob=self.keep_prob)
            token_dropout_mask = tf.nn.dropout(tf.ones([bs, self.word_emb_dim], dtype=tf.float32), keep_prob=self.keep_prob)
            output_token = tf.zeros(shape=(bs,), dtype=tf.int32)
            pred_tokens = []
            logits = []
            probs = []
            for i in range(self.seq_len_limit):
                inp_match, _ = attention(persona_do, cell_out * dropout_mask, self.hidden_size, self.persona_mask, scope='att_match')
                inp_utt, _ = attention(utt_do, cell_out * dropout_mask, self.hidden_size, self.utt_mask, scope='att_utt')

                if i > 0 and self.teacher_forcing:
                    input_token_emb = tf.cond(
                                            tf.logical_and(
                                                self.is_train_ph,
                                                tf.random_uniform(shape=(), maxval=1) <= self.teacher_forcing_rate_ph
                                            ),
                                            lambda: tf.nn.embedding_lookup(self.word_emb, self.y[:, i-1]),
                                            lambda: tf.nn.embedding_lookup(self.word_emb, output_token))
                else:
                    input_token_emb = tf.nn.embedding_lookup(self.word_emb, output_token)

                input_token_emb = token_dropout_mask * input_token_emb

                input = tf.concat([input_token_emb, inp_match, inp_utt], axis=-1)
                #input = tf.concat([input_token_emb, inp_match], axis=-1)

                cell_out, cell_state = decoder_cell(input, cell_state)

                output_proj = tf.layers.dense(cell_out, self.word_emb_dim, activation=tf.nn.tanh,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              name='proj', reuse=tf.AUTO_REUSE)

                output_logits = tf.matmul(output_proj, self.word_emb, transpose_b=True)
                logits.append(output_logits)
                output_probs = tf.nn.softmax(logits[-1])
                probs.append(output_probs)
                output_token = tf.argmax(output_probs, axis=-1)
                pred_tokens.append(output_token)

            self.y_pred = tf.transpose(tf.stack(pred_tokens, axis=0), [1, 0])
            print('y_pred:', self.y_pred)
            self.logits = tf.transpose(tf.stack(logits, axis=0), [1, 0, 2])
            print('logits:', self.logits)

            self.y_probs = tf.transpose(tf.stack(probs, axis=0), [1, 0, 2])
            print('probs:', self.y_probs)

        self.y_ohe = tf.one_hot(self.y, depth=self.vocab_size)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_ohe, logits=self.logits) * self.y_mask
        self.loss = tf.reduce_sum(self.loss) / tf.reduce_sum(self.y_mask)
        self.loss_summary = tf.summary.scalar("loss", self.loss)

    def _init_placeholders(self):
        self.utt_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='utt_ph')
        self.utt_c_ph = tf.placeholder(shape=(None, None, self.char_limit), dtype=tf.int32, name='utt_c_ph')
        self.persona_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='persona_ph')
        self.persona_c_ph = tf.placeholder(shape=(None, None, self.char_limit), dtype=tf.int32, name='persona_c_ph')
        self.y_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='y_ph')

        self.lr_ph = tf.placeholder(dtype=tf.float32, shape=[], name='lr_ph')
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=[], name='keep_prob_ph')
        self.is_train_ph = tf.placeholder_with_default(False, shape=[], name='is_train_ph')
        self.teacher_forcing_rate_ph = tf.placeholder_with_default(0.0, shape=[], name='teacher_forcing_rate_ph')

    def _build_feed_dict(self, persona_tokens, persona_chars, utt_tokens, utt_chars, y=None, mode='train'):
        if y is None:
            feed_dict = {
                self.utt_ph: utt_tokens,
                self.utt_c_ph: utt_chars,
                self.persona_ph: persona_tokens,
                self.persona_c_ph: persona_chars,
                self.y_ph: np.zeros(shape=(len(utt_tokens), self.seq_len_limit))
            }
        elif mode == 'train':
            feed_dict = {
                self.utt_ph: utt_tokens,
                self.utt_c_ph: utt_chars,
                self.persona_ph: persona_tokens,
                self.persona_c_ph: persona_chars,
                self.y_ph: y,
                self.lr_ph: self.learning_rate,
                self.keep_prob_ph: self.keep_prob,
                self.is_train_ph: True,
                self.teacher_forcing_rate_ph: self.teacher_forcing_rate,
            }
        else:
            feed_dict = {
                self.utt_ph: utt_tokens,
                self.utt_c_ph: utt_chars,
                self.persona_ph: persona_tokens,
                self.persona_c_ph: persona_chars,
                self.y_ph: y,
                self.is_train_ph: True,
                self.teacher_forcing_rate_ph: 1.0,
            }

        return feed_dict

    def train_on_batch(self, persona_tokens, persona_chars, utt_tokens, utt_chars, y):
        feed_dict = self._build_feed_dict(persona_tokens, persona_chars, utt_tokens, utt_chars, y, mode='train')
        loss, loss_summary, _ = self.sess.run([self.loss, self.loss_summary, self.train_op], feed_dict=feed_dict)
        if random.randint(0, 99) % 10 == 0:
            print('loss:', loss)

        if self.summary_writer is None:
            self.summary_writer = tf.summary.FileWriter(self.log_path, graph=self.sess.graph)

        self.summary_writer.add_summary(loss_summary, self.step)
        self.step += 1
        return loss

    def __call__(self, persona_tokens, persona_chars, utt_tokens, utt_chars, y=None, *args, **kwargs):
        feed_dict = self._build_feed_dict(persona_tokens, persona_chars, utt_tokens, utt_chars, y, mode='predict')
        y_pred, y_probs = self.sess.run([self.y_pred, self.y_probs], feed_dict=feed_dict)
        return y_pred, y_probs

    def shutdown(self):
        pass
