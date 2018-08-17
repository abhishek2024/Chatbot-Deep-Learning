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
from deeppavlov.core.common.log import get_logger

log = get_logger(__name__)


@register("seq2seq_go_bot_keras")
class Seq2SeqGoalOrientedBotKerasNetwork(KerasModel):

    def __init__(self,
                 hidden_size: int,
                 source_vocab_size: int,
                 target_vocab_size: int,
                 target_start_of_sequence_index: int,
                 target_end_of_sequence_index: int,
                 encoder_embedding_size: int,
                 decoder_embedding_size: int,
                 decoder_embeddings: np.ndarray,
                 knowledge_base_entry_embeddings: np.ndarray = None,
                 kb_attention_hidden_sizes: List[int] = None,
                 source_max_length: int = None,
                 target_max_length: int = None,
                 beam_width: int = 1,
                 model_name: str = "lstm_lstm_model",
                 optimizer: str = "Adam",
                 loss: str = "binary_crossentropy",
                 lear_rate: float = 0.01,
                 lear_rate_decay: float = 0.,
                 **kwargs):

        super().__init__(hidden_size=hidden_size,
                         src_vocab_size=source_vocab_size,
                         tgt_vocab_size=target_vocab_size,
                         tgt_sos_id=target_start_of_sequence_index,
                         tgt_eos_id=target_end_of_sequence_index,
                         encoder_embedding_size=encoder_embedding_size,
                         decoder_embedding_size=decoder_embedding_size,
                         decoder_embeddings=decoder_embeddings,
                         knowledge_base_entry_embeddings=knowledge_base_entry_embeddings,
                         kb_attention_hidden_sizes=kb_attention_hidden_sizes,
                         src_max_length=source_max_length,
                         tgt_max_length=target_max_length,
                         beam_width=beam_width,
                         model_name=model_name,
                         optimizer=optimizer,
                         loss=loss,
                         lear_rate=lear_rate,
                         lear_rate_decay=lear_rate_decay,
                         **kwargs)

        # Parameters required to init model
        params = {"model_name": self.opt.get('model_name'),
                  "optimizer_name": self.opt.get('optimizer'),
                  "loss_name": self.opt.get('loss'),
                  "lear_rate": self.opt.get('lear_rate'),
                  "lear_rate_decay": self.opt.get('lear_rate_decay')}

        self.opt["decoder_embeddings"] = np.asarray(self.opt["decoder_embeddings"])

        self.encoder_model = None
        self.decoder_model = None
        self.model = self.load(**params)

        self._change_not_fixed_params(hidden_size=hidden_size,
                                      src_vocab_size=source_vocab_size,
                                      tgt_vocab_size=target_vocab_size,
                                      tgt_sos_id=target_start_of_sequence_index,
                                      tgt_eos_id=target_end_of_sequence_index,
                                      encoder_embedding_size=encoder_embedding_size,
                                      decoder_embedding_size=decoder_embedding_size,
                                      decoder_embeddings=decoder_embeddings,
                                      knowledge_base_entry_embeddings=knowledge_base_entry_embeddings,
                                      kb_attention_hidden_sizes=kb_attention_hidden_sizes,
                                      src_max_length=source_max_length,
                                      tgt_max_length=target_max_length,
                                      beam_width=beam_width,
                                      model_name=model_name,
                                      optimizer=optimizer,
                                      loss=loss,
                                      lear_rate=lear_rate,
                                      lear_rate_decay=lear_rate_decay,
                                      **kwargs)
        return

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
            "hidden_size",
            "src_vocab_size",
            "tgt_vocab_size",
            "tgt_sos_id",
            "tgt_eos_id",
            "encoder_embedding_size",
            "decoder_embedding_size",
            "optimizer",
            "loss"
        ]
        for param in self.opt.keys():
            if param not in fixed_params:
                self.opt[param] = kwargs.get(param)
        return

    def lstm_lstm_model(self,
                        hidden_size=300,
                        encoder_coef_reg_lstm=0.,
                        encoder_dropout_rate=0.,
                        encoder_rec_dropout_rate=0.,
                        decoder_coef_reg_lstm=0.,
                        decoder_dropout_rate=0.,
                        decoder_rec_dropout_rate=0.,
                        **kwargs) -> List[Model]:

        self._build_encoder(hidden_size,
                            encoder_coef_reg_lstm,
                            encoder_dropout_rate,
                            encoder_rec_dropout_rate)

        self._build_decoder(hidden_size,
                            decoder_coef_reg_lstm,
                            decoder_dropout_rate,
                            decoder_rec_dropout_rate)

        encoder_decoder_model = Model(inputs=[self._encoder_emb_inp,
                                              self._decoder_inp],
                                      outputs=self._train_decoder_outputs)

        self.encoder_model = Model(inputs=self._encoder_emb_inp,
                                   outputs=[self._encoder_state_0,
                                            self._encoder_state_1])

        self.decoder_model = Model(inputs=[self._decoder_inp,
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
                       hidden_size,
                       encoder_coef_reg_lstm,
                       encoder_dropout_rate,
                       encoder_rec_dropout_rate):

        self._encoder_emb_inp = Input(shape=(self.opt["src_max_length"],
                                             self.opt["encoder_embedding_size"]))

        self._encoder_outputs, self._encoder_state_0, self._encoder_state_1 = LSTM(
            hidden_size,
            activation='tanh',
            return_state=True,  # get encoder's last state
            kernel_regularizer=l2(encoder_coef_reg_lstm),
            dropout=encoder_dropout_rate,
            recurrent_dropout=encoder_rec_dropout_rate,
            name="encoder_lstm")(self._encoder_emb_inp)

        return None

    def _build_decoder(self,
                       hidden_size,
                       decoder_coef_reg_lstm,
                       decoder_dropout_rate,
                       decoder_rec_dropout_rate):

        self._decoder_inp = Input(shape=(self.opt["tgt_max_length"],))

        self._decoder_embedder = Embedding(input_dim=self.opt["decoder_embeddings"].shape[0],
                                           output_dim=self.opt["decoder_embedding_size"],
                                           weights=[self.opt["decoder_embeddings"]],
                                           trainable=False)
        _decoder_emb_inp = self._decoder_embedder(self._decoder_inp)

        self._decoder_input_state_0 = Input(shape=(hidden_size,))
        self._decoder_input_state_1 = Input(shape=(hidden_size,))

        decoder_lstm = LSTM(
            hidden_size,
            activation='tanh',
            return_state=True,  # due to teacher forcing, this state is used only for inference
            return_sequences=True,  # to get decoder_n_tokens outputs' representations
            kernel_regularizer=l2(decoder_coef_reg_lstm),
            dropout=decoder_dropout_rate,
            recurrent_dropout=decoder_rec_dropout_rate,
            name="decoder_lstm")

        _train_decoder_outputs, self._train_decoder_state_0, self._train_decoder_state_1 = decoder_lstm(
            _decoder_emb_inp,
            initial_state=[self._encoder_state_0, self._encoder_state_1])

        _infer_decoder_outputs, self._infer_decoder_state_0, self._infer_decoder_state_1 = decoder_lstm(
            _decoder_emb_inp,
            initial_state=[self._decoder_input_state_0, self._decoder_input_state_1])

        decoder_dense = Dense(self.opt["tgt_vocab_size"], name="dense_lstm")  # (batch_size, text_size, tgt_vocab_size)
        self._train_decoder_outputs = decoder_dense(_train_decoder_outputs)
        self._infer_decoder_outputs = decoder_dense(_infer_decoder_outputs)

        return None

    def train_on_batch(self,
                       enc_inputs,
                       dec_inputs,
                       dec_outputs,
                       src_seq_lengths,
                       tgt_seq_lengths,
                       tgt_weights,
                       kb_masks):
        K.set_session(self.sess)
        dec_outputs = self.one_hotter(dec_outputs, vocab_size=self.opt["tgt_vocab_size"])

        metrics_values = self.model.train_on_batch([enc_inputs,
                                                    dec_inputs],
                                                   dec_outputs)
        return metrics_values

    def infer_on_batch(self,
                       enc_inputs,
                       dec_outputs=None):
        K.set_session(self.sess)
        # self.opt["src_max_length"] = max(src_seq_lengths)
        self.opt["tgt_max_length"] = None
        dec_inputs = self._get_decoder_inputs(enc_inputs=enc_inputs)

        if dec_outputs:
            dec_outputs = self.one_hotter(dec_outputs, vocab_size=self.opt["tgt_vocab_size"])
            _encoder_state_0, _encoder_state_1 = self.encoder_model.predict(enc_inputs)
            metrics_values = self.decoder_model.test_on_batch([dec_inputs,
                                                               _encoder_state_0,
                                                               _encoder_state_1],
                                                              dec_outputs)
            return metrics_values
        else:
            _encoder_state_0, _encoder_state_1 = self.encoder_model.predict(enc_inputs)
            predictions = self._probas2onehot(self.decoder_model.predict([dec_inputs,
                                                                          _encoder_state_0,
                                                                          _encoder_state_1]))
            return predictions

    def __call__(self, enc_inputs, src_seq_lengths, kb_masks, prob=False):
        K.set_session(self.sess)
        predictions = np.array(self.infer_on_batch(enc_inputs=enc_inputs))
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
                dec_inputs.append(sample[1:] + self.opt["decoder_embeddings"][self.opt["tgt_eos_id"]])

        return np.asarray(dec_inputs)

    def shutdown(self):
        self.sess.close()

    def reset(self):
        self.sess.close()
