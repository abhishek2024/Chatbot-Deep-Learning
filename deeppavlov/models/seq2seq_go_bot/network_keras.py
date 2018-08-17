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

from typing import List, Tuple
import json
import numpy as np
from keras.layers import Dense, Input, concatenate, Activation, Concatenate, Reshape, Embedding
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM, GRU
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D
from keras.models import Model
from keras.regularizers import l2
from keras.backend import tile
from keras import backend as K

from deeppavlov.core.models.keras_model import KerasModel
from deeppavlov.core.common.registry import register
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.models.tf_model import TFModel
from deeppavlov.core.common.log import get_logger

log = get_logger(__name__)


@register("seq2seq_go_bot_keras")
class Seq2SeqGoalOrientedBotKerasNetwork(KerasModel):

    def __init__(self,
                 target_start_of_sequence_index: int,
                 target_end_of_sequence_index: int,
                 source_vocab_size: int,
                 target_vocab_size: int,
                 src_max_length: int = None, tgt_max_length: int = None,
                 encoder_embedding_size: int = None,
                 decoder_embedding_size: int = None,
                 model_name: str = "lstm_lstm_model",
                 optimizer: str = "Adam", loss: str = "binary_crossentropy",
                 lear_rate: float = 0.01, lear_rate_decay: float = 0.,
                 **kwargs):

        super().__init__(encoder_embedding_size=encoder_embedding_size,
                         decoder_embedding_size=decoder_embedding_size,
                         src_max_length=src_max_length,
                         tgt_max_length=tgt_max_length,
                         model_name=model_name,
                         optimizer=optimizer,
                         loss=loss,
                         lear_rate=lear_rate,
                         lear_rate_decay=lear_rate_decay,
                         **kwargs)  # self.opt = copy(kwargs) initialized in here

        self.opt["encoder_embedding_size"] = encoder_embedding_size
        self.opt["decoder_embedding_size"] = decoder_embedding_size
        self.opt["src_max_length"] = src_max_length
        self.opt["tgt_max_length"] = tgt_max_length
        self.opt["tgt_sos_id"] = target_start_of_sequence_index
        self.opt["tgt_eos_id"] = target_end_of_sequence_index
        self.opt["src_vocab_size"] = source_vocab_size
        self.opt["tgt_vocab_size"] = target_vocab_size

        # Parameters required to init model
        params = {"model_name": self.opt.get('model_name'),
                  "optimizer_name": self.opt.get('optimizer'),
                  "loss_name": self.opt.get('loss'),
                  "lear_rate": self.opt.get('lear_rate'),
                  "lear_rate_decay": self.opt.get('lear_rate_decay')}

        self.model = self.load(**params)

        self._change_not_fixed_params(encoder_embedding_size=encoder_embedding_size,
                                      decoder_embedding_size=decoder_embedding_size,
                                      model_name=model_name,
                                      optimizer=optimizer,
                                      loss=loss,
                                      tgt_sos_id=target_start_of_sequence_index,
                                      tgt_eos_id=target_end_of_sequence_index,
                                      src_vocab_size=source_vocab_size,
                                      tgt_vocab_size=target_vocab_size,
                                      **kwargs)

    def _change_not_fixed_params(self, **kwargs) -> None:
        """
        Change changable parameters from saved model to given ones.

        Args:
            kwargs: dictionary of new parameters

        Returns:
            None
        """
        fixed_params = [
            "model_name",
            "encoder_embedding_size",
            "decoder_embedding_size",
            "encoder_hidden_size",
            "decoder_hidden_size",
            "tgt_sos_id",
            "tgt_eos_id",
            "src_vocab_size",
            "tgt_vocab_size"
        ]
        for param in self.opt.keys():
            if param not in fixed_params:
                self.opt[param] = kwargs.get(param)
        return

    def lstm_lstm_model(self,
                        encoder_hidden_size=300,
                        encoder_coef_reg_lstm=0.,
                        encoder_dropout_rate=0.,
                        encoder_rec_dropout_rate=0.,
                        decoder_hidden_size=300,
                        decoder_coef_reg_lstm=0.,
                        decoder_dropout_rate=0.,
                        decoder_rec_dropout_rate=0., **kwargs) -> List[Model]:

        self._build_encoder(encoder_hidden_size,
                            encoder_coef_reg_lstm,
                            encoder_dropout_rate,
                            encoder_rec_dropout_rate)

        self._build_decoder(encoder_hidden_size,
                            decoder_hidden_size,
                            decoder_coef_reg_lstm,
                            decoder_dropout_rate,
                            decoder_rec_dropout_rate)

        encoder_decoder_model = Model(inputs=[self._encoder_emb_inp,
                                              self._decoder_emb_inp],
                                      outputs=self._train_decoder_outputs)

        self.encoder_model = Model(inputs=self._encoder_emb_inp,
                              outputs=[self._encoder_state_0,
                                       self._encoder_state_1])

        self.decoder_model = Model(inputs=[self._decoder_emb_inp,
                                      self._decoder_input_state_0,
                                      self._decoder_input_state_1],
                              outputs=self._infer_decoder_outputs)

        return encoder_decoder_model

    def one_hotter(self, data, vocab_size):
        """
        Convert given batch of tokenized samples with indexed tokens
        to one-hot representation of tokens

        Args:
            data: list of samples. Each sample is a list of indexes of words in the vocabulary
            vocab_size: size of the vocabulary. Length of one-hot vector

        Returns:
            one-hot representation of given batch
        """
        vocab_matrix = np.eye(vocab_size)

        if type(data) is int:
            one_hotted_data = vocab_matrix[data]
        else:
            one_hotted_data = []
            for sample in data:
                one_hotted_data.append([vocab_matrix[token] for token in sample])

        return np.asarray(one_hotted_data)

    def _build_encoder(self,
                       encoder_hidden_size,
                       encoder_coef_reg_lstm,
                       encoder_dropout_rate,
                       encoder_rec_dropout_rate):

        if self.opt["encoder_embedding_size"] is None:
            self._encoder_emb_inp = Input(shape=(self.opt["src_max_length"],
                                                 self.opt["src_vocab_size"]))
        else:
            self._encoder_emb_inp = Input(shape=(self.opt["src_max_length"],
                                                 self.opt["encoder_embedding_size"]))

        self._encoder_outputs, self._encoder_state_0, self._encoder_state_1 = LSTM(
            encoder_hidden_size,
            activation='tanh',
            return_state=True,  # get encoder's last state
            kernel_regularizer=l2(encoder_coef_reg_lstm),
            dropout=encoder_dropout_rate,
            recurrent_dropout=encoder_rec_dropout_rate,
            name="encoder_lstm")(self._encoder_emb_inp)

    def _build_decoder(self,
                       encoder_hidden_size,
                       decoder_hidden_size,
                       decoder_coef_reg_lstm,
                       decoder_dropout_rate,
                       decoder_rec_dropout_rate):

        if self.opt["decoder_embedding_size"] is None:
            self._decoder_emb_inp = Input(shape=(self.opt["tgt_max_length"],
                                                 self.opt["tgt_vocab_size"]))
        else:
            self._decoder_emb_inp = Input(shape=(self.opt["tgt_max_length"],
                                                 self.opt["decoder_embedding_size"]))

        self._decoder_input_state_0 = Input(shape=(encoder_hidden_size,))
        self._decoder_input_state_1 = Input(shape=(encoder_hidden_size,))

        decoder_lstm = LSTM(
            decoder_hidden_size,
            activation='tanh',
            return_state=True,  # due to teacher forcing, this state is used only for inference
            return_sequences=True,  # to get decoder_n_tokens outputs' representations
            kernel_regularizer=l2(decoder_coef_reg_lstm),
            dropout=decoder_dropout_rate,
            recurrent_dropout=decoder_rec_dropout_rate,
            name="decoder_lstm")

        self._train_decoder_outputs, self._train_decoder_state_0, self._train_decoder_state_1 = decoder_lstm(
            self._decoder_emb_inp,
            initial_state=[self._encoder_state_0, self._encoder_state_1])

        self._infer_decoder_outputs, self._infer_decoder_state_0, self._infer_decoder_state_1 = decoder_lstm(
            self._decoder_emb_inp,
            initial_state=[self._decoder_input_state_0, self._decoder_input_state_1])

        decoder_dense = Dense(self.opt["tgt_vocab_size"], name="dense_lstm")
        self._train_decoder_outputs = decoder_dense(self._train_decoder_outputs)
        self._infer_decoder_outputs = decoder_dense(self._infer_decoder_outputs)

    def train_on_batch(self, enc_inputs, dec_inputs, dec_outputs,
                       src_seq_lengths, tgt_seq_lengths, tgt_weights):
        K.set_session(self.sess)
        # self.opt["src_max_length"] = max(src_seq_lengths)
        # self.opt["tgt_max_length"] = max(tgt_seq_lengths)

        if self.opt["encoder_embedding_size"] is None:
            enc_inputs = self.one_hotter(enc_inputs, self.opt["src_vocab_size"])
        if self.opt["decoder_embedding_size"] is None:
            dec_inputs = self.one_hotter(dec_inputs, self.opt["tgt_vocab_size"])
            dec_outputs = self.one_hotter(dec_outputs, self.opt["tgt_vocab_size"])

        metrics_values = self.model.train_on_batch([enc_inputs,
                                                    dec_inputs],
                                                   dec_outputs)
        return metrics_values

    def infer_on_batch(self, enc_inputs, src_seq_lengths, dec_outputs=None):
        K.set_session(self.sess)
        # self.opt["src_max_length"] = max(src_seq_lengths)
        self.opt["tgt_max_length"] = None
        dec_inputs = self._get_decoder_inputs(enc_inputs=enc_inputs)

        if dec_outputs:
            if self.opt["encoder_embedding_size"] is None:
                enc_inputs = self.one_hotter(enc_inputs, self.opt["src_vocab_size"])
            if self.opt["decoder_embedding_size"] is None:
                dec_inputs = self.one_hotter(dec_inputs, self.opt["tgt_vocab_size"])
                dec_outputs = self.one_hotter(dec_outputs, self.opt["tgt_vocab_size"])

            _encoder_state_0, _encoder_state_1 = self.encoder_model.predict(enc_inputs)
            metrics_values = self.decoder_model.test_on_batch([dec_inputs,
                                                               _encoder_state_0,
                                                               _encoder_state_1],
                                                              dec_outputs)
            return metrics_values
        else:
            if self.opt["encoder_embedding_size"] is None:
                enc_inputs = self.one_hotter(enc_inputs, self.opt["src_vocab_size"])
            if self.opt["decoder_embedding_size"] is None:
                dec_inputs = self.one_hotter(dec_inputs, self.opt["tgt_vocab_size"])

            _encoder_state_0, _encoder_state_1 = self.encoder_model.predict(enc_inputs)
            predictions = self._probas2onehot(self.decoder_model.predict([dec_inputs,
                                                                          _encoder_state_0,
                                                                          _encoder_state_1]))
            return predictions

    def __call__(self, enc_inputs, src_seq_lengths, prob=False):
        K.set_session(self.sess)
        predictions = np.array(self.infer_on_batch(enc_inputs=enc_inputs,
                                                   src_seq_lengths=src_seq_lengths))
        return predictions

    def _probas2onehot(self, data):
        text_data = []

        for sample in data:
            text_sample = []
            for token in sample:
                text_sample.append(np.argmax(token))
            text_data.append(text_sample)

        return np.asarray(text_data)

    def _get_decoder_inputs(self, enc_inputs):
        dec_inputs = []
        if type(enc_inputs[0][0]) is int:
            for sample in enc_inputs:
                dec_inputs.append(sample[1:] + [self.opt["tgt_eos_id"]])
        else:
            for sample in enc_inputs:
                dec_inputs.append(sample[1:] + self.one_hotter(self.opt["tgt_eos_id"], self.opt["tgt_vocab_size"]))
        return dec_inputs

    def shutdown(self):
        self.sess.close()

    def reset(self):
        self.sess.close()
