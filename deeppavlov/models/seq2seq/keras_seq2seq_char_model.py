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

from typing import List, Tuple, Union, Optional
import numpy as np
from logging import getLogger
from overrides import overrides
from copy import deepcopy

from keras.layers import Dense, Input, Embedding
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.regularizers import l2

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.models.classifiers.keras_classification_model import KerasClassificationModel

log = getLogger(__name__)


@register("keras_seq2seq_char_model")
class KerasSeq2SeqCharModel(KerasClassificationModel):
    """
    Class implements Keras model for seq2seq task on char-level

    Args:
        hidden_size: size of the hidden layer of encoder and decoder
        src_vocab_size: vocabulary size of source sequences
        tgt_vocab_size: vocabulary size of target sequences
        src_pad_id: index of padding special token in source vocabulary
        tgt_pad_id: index of padding special token in target vocabulary
        tgt_bos_id: index of begin-of-sequence special char in target vocabulary
        tgt_eos_id: index of end-of-sequence special char in target vocabulary
        encoder_embedding_size: embedding size of encoder's embedder
        decoder_vocab: decoder's vocab component
        src_max_length: maximal char length of source sequence
        tgt_max_length: maximal char length of target sequence
        model_name: string name of particular method of this class that builds connection between encoder and decoder
        encoder_name: string name of particular method of this class that builds encoder model
        decoder_name: string name of particular method of this class that builds decoder model
        optimizer: string name of optimizer from keras.optimizers
        loss: string name of loss from keras.losses
        learning_rate: learning rate for optimizer
        learning_rate_decay: learning rate decay for optimizer
        restore_lr: whether to reinitialize learning rate value  \
            within the final stored in model_opt.json (if model was loaded)
        encoder_coef_reg_lstm: coefficient for L2 kernel regularizer of encoder LSTM layer
        encoder_dropout_rate: dropout rate for encoder LSTM layer
        encoder_rec_dropout_rate: recurrent dropout rate for encoder LSTM layer
        decoder_coef_reg_lstm: coefficient for L2 kernel regularizer of decoder LSTM layer
        decoder_dropout_rate: dropout rate for decoder LSTM layer
        decoder_rec_dropout_rate: recurrent dropout rate for decoder LSTM layer
        **kwargs: additional arguments

    Attributes:
        opt: dictionary with model parameters
        decoder_embedder: decoder's embedder component
        decoder_vocab: decoder's vocab component
        encoder_model: Keras model for encoder (for infering)
        decoder_model: Keras model for decoder (for infering)
        model: Keras model of connected encoder and decoder (for training)

    """
    def __init__(self,
                 hidden_size: int,
                 src_pad_id: int,
                 tgt_pad_id: int,
                 tgt_bos_id: int,
                 tgt_eos_id: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 decoder_vocab: Component,
                 encoder_embedding_size: Optional[int] = None,
                 decoder_embedding_size: Optional[int] = None,
                 src_max_length: Optional[int] = None,
                 tgt_max_length: Optional[int] = None,
                 model_name: str = "encoder_decoder_model",
                 encoder_name: str = "lstm_encoder_model",
                 decoder_name: str = "lstm_decoder_model",
                 optimizer: str = "Adam",
                 loss: str = "categorical_crossentropy",
                 learning_rate: float = 0.01,
                 learning_rate_decay: float = 0.,
                 restore_lr: bool = False,
                 encoder_coef_reg_lstm: float = 0.,
                 encoder_dropout_rate: float = 0.,
                 encoder_rec_dropout_rate: float = 0.,
                 decoder_coef_reg_lstm: float = 0.,
                 decoder_dropout_rate: float = 0.,
                 decoder_rec_dropout_rate: float = 0.,
                 self_att_enc_hid: int = None,
                 self_att_enc_out: int = None,
                 self_att_dec_hid: int = None,
                 self_att_dec_out: int = None,
                 **kwargs) -> None:
        """
        Initialize models for training and infering using parameters from config.
        """
        given_opt = {"hidden_size": hidden_size,
                     "src_vocab_size": src_vocab_size,
                     "tgt_vocab_size": tgt_vocab_size,
                     "src_pad_id": src_pad_id,
                     "tgt_pad_id": tgt_pad_id,
                     "tgt_bos_id": tgt_bos_id,
                     "tgt_eos_id": tgt_eos_id,
                     "encoder_embedding_size": encoder_embedding_size,
                     "decoder_embedding_size": decoder_embedding_size,
                     "src_max_length": src_max_length,
                     "tgt_max_length": tgt_max_length,
                     "model_name": model_name,
                     "encoder_name": model_name,
                     "decoder_name": model_name,
                     "optimizer": optimizer,
                     "loss": loss,
                     "learning_rate": learning_rate,
                     "learning_rate_decay": learning_rate_decay,
                     "restore_lr": restore_lr,
                     "encoder_coef_reg_lstm": encoder_coef_reg_lstm,
                     "encoder_dropout_rate": encoder_dropout_rate,
                     "encoder_rec_dropout_rate": encoder_rec_dropout_rate,
                     "decoder_coef_reg_lstm": decoder_coef_reg_lstm,
                     "decoder_dropout_rate": decoder_dropout_rate,
                     "decoder_rec_dropout_rate": decoder_rec_dropout_rate,
                     "self_att_enc_hid": self_att_enc_hid,
                     "self_att_enc_out": self_att_enc_out,
                     "self_att_dec_hid": self_att_dec_hid,
                     "self_att_dec_out": self_att_dec_out,
                     **kwargs}

        self.opt = deepcopy(given_opt)

        self.encoder_method = getattr(self, encoder_name, None)
        self.decoder_method = getattr(self, decoder_name, None)

        # calling init of KerasModel (not KerasClassificationModel)
        super(KerasClassificationModel, self).__init__(**given_opt)

        self.decoder_vocab = decoder_vocab

        self.encoder_model = None
        self.decoder_model = None
        self._encoder_inp = None
        self._encoder_state = None
        self._decoder_inp = None
        self._decoder_infer_inp = None
        self._decoder_input_state = None
        self._train_decoder_state = None
        self._infer_decoder_state = None
        self._train_decoder_outputs = None
        self._infer_decoder_outputs = None

        self.load(model_name=model_name)

        if restore_lr:
            learning_rate = self.opt.get("final_learning_rate", learning_rate)

        self.model = self.compile(self.model, optimizer_name=optimizer, loss_name=loss,
                                  learning_rate=learning_rate, learning_rate_decay=learning_rate_decay)

        self.encoder_model = self.compile(self.encoder_model, optimizer_name=optimizer, loss_name=loss,
                                          learning_rate=learning_rate, learning_rate_decay=learning_rate_decay)
        self.decoder_model = self.compile(self.decoder_model, optimizer_name=optimizer, loss_name=loss,
                                          learning_rate=learning_rate, learning_rate_decay=learning_rate_decay)

        self._change_not_fixed_params(**given_opt)

        summary = ['Model was successfully initialized!', 'Model summary:']
        self.model.summary(print_fn=summary.append)
        log.info('\n'.join(summary))

    @overrides
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
            "tgt_bos_id",
            "tgt_eos_id",
            "encoder_embedding_size",
            "decoder_embedding_size",
            "optimizer",
            "loss",
            "self_att_enc_hid",
            "self_att_enc_out",
            "self_att_dec_hid",
            "self_att_dec_out"
        ]
        for param in self.opt.keys():
            if param not in fixed_params:
                self.opt[param] = kwargs.get(param)
        return

    def encoder_decoder_model(self, **kwargs) -> Model:
        """
        Build keras models for training and infering

        Args:
            **kwargs: parameters of model from config

        Returns:
            keras model for training, also initializes encoder and decoder model separately for infering
        """
        self.encoder_method(**kwargs)

        self.decoder_method(**kwargs)

        encoder_decoder_model = Model(inputs=[self._encoder_inp,
                                              self._decoder_inp],
                                      outputs=self._train_decoder_outputs)
        self.encoder_model = Model(inputs=self._encoder_inp,
                                   outputs=self._encoder_state)

        if isinstance(self._decoder_input_state, list):
            self.decoder_model = Model(inputs=[self._decoder_infer_inp] + self._decoder_input_state,
                                       outputs=[self._infer_decoder_outputs] + self._infer_decoder_state)
        else:
            self.decoder_model = Model(inputs=[self._decoder_infer_inp, self._decoder_input_state],
                                       outputs=[self._infer_decoder_outputs, self._infer_decoder_state])
        return encoder_decoder_model

    @overrides
    def train_on_batch(self, x: Tuple[List[int]], y: Tuple[List[int]], **kwargs) -> Union[float, List[float]]:
        """
        Train the self.model on the given batch using teacher forcing

        Args:
            x: list of input samples where each sample is a list of indices of chars
            y: list of output samples where each sample is a list of indices of chars
            kwargs: additional arguments

        Returns:
            metrics values on the given batch
        """
        pad_enc_inputs = np.array(x)
        pad_dec_inputs = np.array([[np.eye(self.opt["tgt_vocab_size"])[self.opt["tgt_bos_id"]]] +
                                   list(sample) for sample in y])  # (bs, max_ts + 1, dec_emb_size)
        pad_onehot_dec_outputs = np.array([list(sample) + [np.eye(self.opt["tgt_vocab_size"])[self.opt["tgt_pad_id"]]]
                                           for sample in y])  # (bs, max_ts + 1, dec_emb_size)

        metrics_values = self.model.train_on_batch([pad_enc_inputs,
                                                    pad_dec_inputs],
                                                   pad_onehot_dec_outputs)
        return metrics_values

    @overrides
    def infer_on_batch(self, x: List[List[int]], **kwargs) -> List[List[int]]:
        """
        Infer self.encoder_model and self.decoder_model on the given data (no teacher forcing)

        Args:
            x: list of input samples where each sample is a list of indices of chars
            **kwargs: additional arguments

        Returns:
            list of decoder predictions where each prediction is a list of indices of chars
        """
        pad_enc_inputs = np.array(x)
        encoder_state = self.encoder_model.predict(pad_enc_inputs)

        predicted_batch = []
        for i in range(len(x)):  # batch size
            predicted_sample = []

            current_char = np.eye(self.opt["tgt_vocab_size"])[self.opt["tgt_bos_id"]]
            end_of_sequence = False
            if isinstance(encoder_state, list):
                state = [np.array([encoder_state[0][i]]), np.array([encoder_state[1][i]])]
            else:
                state = [np.array([encoder_state[i]])]

            while not end_of_sequence:
                output = self.decoder_model.predict([np.array([[current_char]])] + state)
                char_probas = output[0]
                state = output[1:]
                current_char = self._probas2ids(char_probas)[0][0]  # index!
                if (current_char == self.opt["tgt_eos_id"] or
                        current_char == self.opt["tgt_pad_id"] or
                        len(predicted_sample) == self.opt["tgt_max_length"]):
                    end_of_sequence = True
                else:
                    predicted_sample.append(current_char)
                    current_char = np.eye(self.opt["tgt_vocab_size"])[current_char]  # back to one-hot for next iter

            predicted_batch.append(predicted_sample)

        return predicted_batch

    @overrides
    def __call__(self, x: List[List[int]], **kwargs) -> np.ndarray:
        """
        Infer self.encoder_model and self.decoder_model on the given data (no teacher forcing)

        Args:
            x: list of input samples where each sample is a list of indices of chars
            **kwargs: additional arguments

        Returns:
            array of decoder predictions where each prediction is a list of indices of chars
        """
        predictions = np.array(self.infer_on_batch(x))
        return predictions

    @staticmethod
    def _probas2ids(data: List[List[np.ndarray]]) -> np.ndarray:
        """
        Convert vectors of probabilities distribution of chars in the vocabulary \
        or one-hot token representations in the vocabulary  \
        to corresponding token ids

        Args:
            data: list of samples where each sample is a list of np.ndarray of vocabulary size

        Returns:
            samples where each sample is a list of char's id
        """
        ids_data = np.asarray([[np.argmax(char) for char in sample] for sample in data])

        return ids_data

    @staticmethod
    def _ids2onehot(data: Union[List[List[int]], Tuple[List[int]], np.ndarray], vocab_size: int) -> np.ndarray:
        """
        Convert token ids to one-hot representations in vocabulary of size vocab_size

        Args:
            data: list of samples where each sample is a list of char's id
            vocab_size: size of the vocabulary

        Returns:
            samples where each sample is a list of np.ndarrays of vocabulary size
        """
        onehot = np.eye(vocab_size)
        onehot_data = np.asarray([[onehot[char] for char in sample] for sample in data])

        return onehot_data

    def shutdown(self):
        self.sess.close()

    def reset(self):
        self.sess.close()

    def destroy(self):
        self.decoder_embedder.destroy()
        del self.model
        del self.encoder_model
        del self.decoder_model
        del self.decoder_vocab

    def lstm_encoder_model(self,
                           hidden_size: int,
                           encoder_coef_reg_lstm: float,
                           encoder_dropout_rate: float,
                           encoder_rec_dropout_rate: float,
                           **kwargs) -> None:
        """
        Initialize encoder layers for LSTM encoder

        Args:
            hidden_size: size of the hidden layer of encoder and decoder
            encoder_coef_reg_lstm: coefficient for L2 kernel regularizer of encoder LSTM layer
            encoder_dropout_rate: dropout rate for encoder LSTM layer
            encoder_rec_dropout_rate: recurrent dropout rate for encoder LSTM layer

        Returns:
            None
        """
        if self.opt["encoder_embedding_size"] is None:
            self._encoder_inp = Input(shape=(None, self.opt["src_vocab_size"]))
            _encoder_emb_inp = self._encoder_inp
        else:
            self._encoder_inp = Input(shape=(self.opt["src_max_length"],))
            _encoder_emb_inp = Embedding(input_dim=self.opt["src_vocab_size"],
                                         output_dim=self.opt["encoder_embedding_size"],
                                         input_length=self.opt["src_max_length"])(self._encoder_inp)

        _, _encoder_state_h, _encoder_state_c = LSTM(
            hidden_size,
            activation='tanh',
            return_sequences=True,
            return_state=True,
            kernel_regularizer=l2(encoder_coef_reg_lstm),
            dropout=encoder_dropout_rate,
            recurrent_dropout=encoder_rec_dropout_rate)(_encoder_emb_inp)

        self._encoder_state = [_encoder_state_h, _encoder_state_c]

        return None

    def lstm_decoder_model(self,
                           hidden_size: int,
                           decoder_coef_reg_lstm: float,
                           decoder_dropout_rate: float,
                           decoder_rec_dropout_rate: float,
                           **kwargs) -> None:
        """
        Initialize decoder layers for LSTM decoder

        Args:
            hidden_size: size of the hidden layer of encoder and decoder
            decoder_coef_reg_lstm: coefficient for L2 kernel regularizer of decoder LSTM layer
            decoder_dropout_rate: dropout rate for decoder LSTM layer
            decoder_rec_dropout_rate: recurrent dropout rate for decoder LSTM layer

        Returns:
            None
        """
        if self.opt["decoder_embedding_size"] is None:
            self._decoder_inp = Input(shape=(None, self.opt["tgt_vocab_size"]))
            _decoder_emb_inp = self._decoder_inp
            self._decoder_infer_inp = Input(shape=(1, self.opt["tgt_vocab_size"]))
            _decoder_infer_emb_inp = self._decoder_infer_inp
        else:
            self._decoder_inp = Input(shape=(self.opt["tgt_max_length"],))
            _decoder_emb_inp = Embedding(input_dim=self.opt["tgt_vocab_size"],
                                         output_dim=self.opt["decoder_embedding_size"])(self._decoder_inp)
            self._decoder_infer_inp = Input(shape=(1,))
            _decoder_infer_emb_inp = Embedding(input_dim=self.opt["tgt_vocab_size"],
                                               output_dim=self.opt["decoder_embedding_size"])(self._decoder_infer_inp)

        self._decoder_input_state = [Input(shape=(hidden_size,)),  # h state
                                     Input(shape=(hidden_size,))]  # c state

        decoder_cell = LSTM(
            hidden_size,
            activation='tanh',
            return_sequences=True,
            kernel_regularizer=l2(decoder_coef_reg_lstm),
            dropout=decoder_dropout_rate,
            recurrent_dropout=decoder_rec_dropout_rate,
            return_state=True)

        _train_decoder_outputs, train_state_h, train_state_c = decoder_cell(
            _decoder_emb_inp,
            initial_state=self._encoder_state)
        self._train_decoder_state = [train_state_h, train_state_c]

        _infer_decoder_outputs, infer_state_h, infer_state_c = decoder_cell(
            _decoder_infer_emb_inp,
            initial_state=self._decoder_input_state)
        self._infer_decoder_state = [infer_state_h, infer_state_c]

        decoder_dense = Dense(self.opt["tgt_vocab_size"], name="dense_", activation="softmax")
        self._train_decoder_outputs = decoder_dense(_train_decoder_outputs)
        self._infer_decoder_outputs = decoder_dense(_infer_decoder_outputs)

        return None
