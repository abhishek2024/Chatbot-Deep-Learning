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

from typing import List, Tuple, Union
import json
import numpy as np
import scipy.sparse as sp
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
from deeppavlov.core.models.component import Component
from deeppavlov.core.layers.keras_layers import masking_sequences

log = get_logger(__name__)


@register("keras_seq2seq_token_model")
class KerasSeq2SeqTokenModel(KerasModel):
    """
    Class implements Keras model for seq2seq task

    Args:
        hidden_size: size of the hidden layer of encoder and decoder
        source_vocab_size: vocabulary size of source sequences
        target_vocab_size: vocabulary size of target sequences
        target_start_of_sequence_index: index of start-of-sequence special token in target vocabulary
        target_end_of_sequence_index: index of end-of-sequence special token in target vocabulary
        encoder_embedding_size: embedding size of encoder's embedder
        decoder_embedder: decoder's embedder component
        decoder_vocab: decoder's vocab component
        source_max_length: maximal token length of source sequence
        target_max_length: maximal token length of target sequence
        model_name: string name of particular method of this class that builds seq2seq model
        optimizer: string name of optimizer from keras.optimizers
        loss: string name of loss from keras.losses
        lear_rate learning rate for optimizer
        lear_rate_decay: learning rate decay for optimizer
        **kwargs: additional arguments

    Attributes:
        opt: dictionary with model parameters
        decoder_embedder: decoder's embedder component
        decoder_vocab: decoder's vocab component
        encoder_model:
        decoder_model:
        model:

    """
    def __init__(self,
                 hidden_size: int,
                 target_vocab_size: int,
                 target_padding_index: int,
                 target_start_of_sequence_index: int,
                 target_end_of_sequence_index: int,
                 encoder_embedding_size: int,
                 decoder_embedder: Component,
                 decoder_vocab: Component,
                 source_max_length: int = None,
                 target_max_length: int = None,
                 model_name: str = "lstm_lstm_model",
                 optimizer: str = "Adam",
                 loss: str = "binary_crossentropy",
                 lear_rate: float = 0.01,
                 lear_rate_decay: float = 0.,
                 restore_lr: bool = False,
                 **kwargs):
        """
        Initialize model using parameters from config.
        """
        decoder_embedding_size = kwargs.pop("decoder_embedding_size", decoder_embedder.dim)

        super().__init__(hidden_size=hidden_size,
                         tgt_vocab_size=target_vocab_size,
                         tgt_pad_id=target_padding_index,
                         tgt_sos_id=target_start_of_sequence_index,
                         tgt_eos_id=target_end_of_sequence_index,
                         encoder_embedding_size=encoder_embedding_size,
                         decoder_embedding_size=decoder_embedding_size,
                         src_max_length=source_max_length,
                         tgt_max_length=target_max_length,
                         model_name=model_name,
                         optimizer=optimizer,
                         loss=loss,
                         lear_rate=lear_rate,
                         lear_rate_decay=lear_rate_decay,
                         restore_lr=restore_lr,
                         **kwargs)

        # Parameters required to init model
        params = {"model_name": self.opt.get('model_name'),
                  "optimizer_name": self.opt.get('optimizer'),
                  "loss_name": self.opt.get('loss'),
                  "lear_rate": self.opt.get('lear_rate'),
                  "lear_rate_decay": self.opt.get('lear_rate_decay')}

        self.decoder_embedder = decoder_embedder
        self.decoder_vocab = decoder_vocab

        self.encoder_model = None
        self.decoder_model = None

        self.model = self.load(model_name=model_name)

        if restore_lr:
            lear_rate = self.opt.get("final_lear_rate", lear_rate)

        self.model = self.compile(self.model, optimizer_name=optimizer, loss_name=loss,
                                  lear_rate=lear_rate, lear_rate_decay=lear_rate_decay)

        self.encoder_model = self.compile(self.encoder_model, optimizer_name=optimizer, loss_name=loss,
                                          lear_rate=lear_rate, lear_rate_decay=lear_rate_decay)
        self.decoder_model = self.compile(self.decoder_model, optimizer_name=optimizer, loss_name=loss,
                                          lear_rate=lear_rate, lear_rate_decay=lear_rate_decay)

        self._change_not_fixed_params(hidden_size=hidden_size,
                                      tgt_vocab_size=target_vocab_size,
                                      tgt_pad_id=target_padding_index,
                                      tgt_sos_id=target_start_of_sequence_index,
                                      tgt_eos_id=target_end_of_sequence_index,
                                      encoder_embedding_size=encoder_embedding_size,
                                      decoder_embedding_size=decoder_embedding_size,
                                      src_max_length=source_max_length,
                                      tgt_max_length=target_max_length,
                                      model_name=model_name,
                                      optimizer=optimizer,
                                      loss=loss,
                                      lear_rate=lear_rate,
                                      lear_rate_decay=lear_rate_decay,
                                      restore_lr=restore_lr,
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

    def texts2decoder_embeddings(self, sentences: List[List[str]],
                                 text_size: int, embedding_size: int = None,
                                 return_lengths=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert texts to vector representations using decoder_embedder \
        and padding up to self.opt["tgt_max_length"] tokens

        Args:
            sentences: list of lists of tokens

        Returns:
            (array of embedded texts, list of sentences lengths)
        """
        pad = np.zeros(embedding_size)
        embeddings_batch = self.decoder_embedder([sen[:text_size] for sen in sentences])
        embeddings_batch = [tokens + [pad] * (text_size - len(tokens))
                            for tokens in embeddings_batch]

        embeddings_batch = np.asarray(embeddings_batch)
        if return_lengths:
            lengths = np.array([min(len(sen), text_size) for sen in sentences], dtype='int')
            return embeddings_batch, lengths

        return embeddings_batch

    def pad_texts(self, sentences: Union[List[List[np.ndarray]], List[List[int]]],
                  text_size: int, embedding_size: int = None,
                  padding_token_id: int = 0, return_lengths=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Cut and pad tokenized sequences (each sample is a list of indexed tokens or list of embedded tokens) \
        up to text_size tokens with zeros (or array of zeros of size embedding_size in case of embedded tokens)

        Args:
            sentences: list of lists of indexed or embedded tokens
            text_size: number of tokens to pad
            embedding_size: embedding size if sample is a list of embedded tokens
            padding_token_id:
            return_lengths:

        Returns:
            (array of embedded texts, list of sentences lengths)
        """
        if type(sentences[0][0]) is int:
            pad = padding_token_id
        else:
            pad = np.zeros(embedding_size)

        cutted_batch = [sen[:text_size] for sen in sentences]
        cutted_batch = [list(tokens) + [pad] * (text_size - len(tokens)) for tokens in cutted_batch]
        if return_lengths:
            lengths = np.array([min(len(sen), text_size) for sen in sentences], dtype='int')
            return np.asarray(cutted_batch), lengths

        return np.asarray(cutted_batch)

    def gru_gru_model(self,
                      hidden_size: int = 300,
                      encoder_coef_reg_lstm: float = 0.,
                      encoder_dropout_rate: float = 0.,
                      encoder_rec_dropout_rate: float = 0.,
                      decoder_coef_reg_lstm: float = 0.,
                      decoder_dropout_rate: float = 0.,
                      decoder_rec_dropout_rate: float = 0.,
                      **kwargs) -> Model:
        """
        Build keras models for training and infering

        Args:
            hidden_size: size of the hidden layer of encoder and decoder
            encoder_coef_reg_lstm: coefficient for L2 kernel regularizer of encoder LSTM layer
            encoder_dropout_rate: dropout rate for encoder LSTM layer
            encoder_rec_dropout_rate: recurrent dropout rate for encoder LSTM layer
            decoder_coef_reg_lstm: coefficient for L2 kernel regularizer of decoder LSTM layer
            decoder_dropout_rate: dropout rate for decoder LSTM layer
            decoder_rec_dropout_rate: recurrent dropout rate for decoder LSTM layer
            **kwargs: additional arguments

        Returns:
            keras model for training
        """

        self._build_encoder(hidden_size,
                            encoder_coef_reg_lstm,
                            encoder_dropout_rate,
                            encoder_rec_dropout_rate)

        self._build_decoder(hidden_size,
                            decoder_coef_reg_lstm,
                            decoder_dropout_rate,
                            decoder_rec_dropout_rate)

        encoder_decoder_model = Model(inputs=[self._encoder_emb_inp,
                                              self._encoder_inp_lengths,
                                              self._decoder_emb_inp,
                                              self._decoder_inp_lengths],
                                      outputs=self._train_decoder_outputs)

        self.encoder_model = Model(inputs=[self._encoder_emb_inp,
                                           self._encoder_inp_lengths],
                                   outputs=self._encoder_state)

        self.decoder_model = Model(inputs=[self._decoder_emb_inp,
                                           self._decoder_inp_lengths,
                                           self._decoder_input_state],
                                   outputs=[self._infer_decoder_outputs,
                                            self._infer_decoder_state])

        return encoder_decoder_model

    def _build_encoder(self,
                       hidden_size: int,
                       encoder_coef_reg_lstm: float,
                       encoder_dropout_rate: float,
                       encoder_rec_dropout_rate: float) -> None:
        """
        Initialize encoder layers

        Args:
            hidden_size: size of the hidden layer of encoder and decoder
            encoder_coef_reg_lstm: coefficient for L2 kernel regularizer of encoder LSTM layer
            encoder_dropout_rate: dropout rate for encoder LSTM layer
            encoder_rec_dropout_rate: recurrent dropout rate for encoder LSTM layer

        Returns:
            None
        """

        self._encoder_emb_inp = Input(shape=(self.opt["src_max_length"],
                                             self.opt["encoder_embedding_size"]))

        self._encoder_inp_lengths = Input(shape=(2,), dtype='int32')

        _encoder_outputs, _encoder_state = GRU(
            hidden_size,
            activation='tanh',
            return_state=True,  # get encoder's last state
            return_sequences=True,
            kernel_regularizer=l2(encoder_coef_reg_lstm),
            dropout=encoder_dropout_rate,
            recurrent_dropout=encoder_rec_dropout_rate,
            name="encoder_gru")(self._encoder_emb_inp)

        self._encoder_state = masking_sequences(_encoder_outputs, self._encoder_inp_lengths)
        return None

    def _build_decoder(self,
                       hidden_size: int,
                       decoder_coef_reg_lstm: float,
                       decoder_dropout_rate: float,
                       decoder_rec_dropout_rate: float) -> None:
        """
        Initialize decoder layers

        Args:
            hidden_size: size of the hidden layer of encoder and decoder
            decoder_coef_reg_lstm: coefficient for L2 kernel regularizer of decoder LSTM layer
            decoder_dropout_rate: dropout rate for decoder LSTM layer
            decoder_rec_dropout_rate: recurrent dropout rate for decoder LSTM layer

        Returns:
            None
        """

        self._decoder_emb_inp = Input(shape=(None, self.opt["decoder_embedding_size"]))
        self._decoder_inp_lengths = Input(shape=(2,), dtype='int32')

        self._decoder_input_state = Input(shape=(hidden_size,))

        decoder_gru = GRU(
            hidden_size,
            activation='tanh',
            return_state=True,  # due to teacher forcing, this state is used only for inference
            return_sequences=True,  # to get decoder_n_tokens outputs' representations
            kernel_regularizer=l2(decoder_coef_reg_lstm),
            dropout=decoder_dropout_rate,
            recurrent_dropout=decoder_rec_dropout_rate,
            name="decoder_gru")

        _train_decoder_outputs, _train_decoder_state = decoder_gru(
            self._decoder_emb_inp,
            initial_state=self._encoder_state)
        self._train_decoder_state = masking_sequences(_train_decoder_state, self._decoder_inp_lengths)

        _infer_decoder_outputs, self._infer_decoder_state = decoder_gru(
            self._decoder_emb_inp,
            initial_state=self._decoder_input_state)

        decoder_dense = Dense(self.opt["tgt_vocab_size"], name="dense_gru", activation="softmax")
        self._train_decoder_outputs = decoder_dense(_train_decoder_outputs)
        self._infer_decoder_outputs = decoder_dense(_infer_decoder_outputs)

        return None

    def train_on_batch(self, *args: Tuple[List[int], List[int]], **kwargs) -> Union[float, List[float]]:
        """
        Train the self.model on the given batch using teacher forcing

        Args:
            args: list of tokenized text samples
            kwargs: additional arguments

        Returns:
            metrics values on the given batch
        """
        pad_emb_enc_inputs, enc_inp_lengths = self.pad_texts(args[0],
                                                             text_size=self.opt["src_max_length"],
                                                             embedding_size=self.opt["encoder_embedding_size"],
                                                             return_lengths=True)
        dec_inputs = [[self.opt["tgt_sos_id"]] +
                      list(sample) + [self.opt["tgt_eos_id"]]
                      for sample in args[1]]  # (bs, ts + 2) of integers (tokens ids)
        text_dec_inputs = self.decoder_vocab(dec_inputs)

        pad_emb_dec_inputs, dec_inp_lengths = self.texts2decoder_embeddings(
            text_dec_inputs,
            text_size=self.opt["tgt_max_length"],
            embedding_size=self.opt["decoder_embedding_size"],
            return_lengths=True)

        # pad_emb_dec_outputs = self.texts2decoder_embeddings(args[1],
        #                                                     text_size=self.opt["tgt_max_length"],
        #                                                     embedding_size=self.opt["decoder_embedding_size"])
        pad_dec_outputs = self.pad_texts(args[1],
                                         text_size=self.opt["tgt_max_length"],
                                         embedding_size=None,
                                         padding_token_id=self.opt["tgt_pad_id"])
        pad_onehot_dec_outputs = self._ids2onehot(pad_dec_outputs, vocab_size=self.opt["tgt_vocab_size"])


        metrics_values = self.model.train_on_batch([pad_emb_enc_inputs,
                                                    np.hstack((np.arange(len(enc_inp_lengths)).reshape(-1, 1),
                                                               enc_inp_lengths.reshape(-1, 1))),
                                                    pad_emb_dec_inputs,
                                                    np.hstack((np.arange(len(dec_inp_lengths)).reshape(-1, 1),
                                                               dec_inp_lengths.reshape(-1, 1)))],
                                                   pad_onehot_dec_outputs)
        return metrics_values

    def infer_on_batch(self, *args: Tuple[Tuple[np.ndarray]], **kwargs) -> List[List[int]]:
        """
        Infer self.encoder_model and self.decoder_model  on the given data (no teacher forcing)

        Args:
            *args: encoder inputs (tokenized embedded sentences)
            **kwargs: additional arguments

        Returns:
            tokenized indexed decoder predictions
        """
        batch = args[0][0]
        pad_emb_enc_inputs, enc_inp_lengths = self.pad_texts(batch,
                                                             text_size=self.opt["src_max_length"],
                                                             embedding_size=self.opt["encoder_embedding_size"],
                                                             return_lengths=True)
        encoder_state = self.encoder_model.predict([pad_emb_enc_inputs,
                                                    np.hstack((np.arange(len(enc_inp_lengths)).reshape(-1, 1),
                                                               enc_inp_lengths.reshape(-1, 1)))])

        predicted_batch = []
        for i in range(len(batch)):  # batch size
            predicted_sample = []

            current_token = self.decoder_embedder([[self.decoder_vocab[self.opt["tgt_sos_id"]]]])[0][0]  # (300,)
            end_of_sequence = False
            state = encoder_state[i].reshape((1, -1))

            while not end_of_sequence:
                token_probas, state = self.decoder_model.predict([np.array([[current_token]]),
                                                                  np.array([[0, 1]], dtype='int').reshape(1, 2),
                                                                  state])
                current_token_id = self._probas2ids(token_probas)[0][0]
                current_token = self.decoder_embedder(self.decoder_vocab([[current_token_id]]))[0][0]
                predicted_sample.append(current_token_id)
                if (current_token_id == self.opt["tgt_eos_id"] or
                        current_token_id == self.opt["tgt_pad_id"] or
                        len(predicted_sample) == self.opt["tgt_max_length"]):
                    end_of_sequence = True

            predicted_batch.append(predicted_sample)

        return predicted_batch

    def __call__(self, *args: Tuple[Tuple[np.ndarray]], **kwargs) -> np.ndarray:
        """
        Infer self.encoder_model and self.decoder_model on the given data

        Args:
            *args: encoder inputs (tokenized embedded sentences)
            **kwargs: additional arguments

        Returns:
            tokenized indexed decoder predictions
        """
        predictions = np.array(self.infer_on_batch(args))
        return predictions

    def _probas2ids(self, data: List[List[np.ndarray]]) -> np.ndarray:
        """
        Convert vectors of probabilities distribution of tokens in the vocabulary \
        or one-hot token representations in the vocabulary  \
        to corresponding token ids

        Args:
            data: list of tokenized samples where each sample is a list of np.ndarray of vocabulary size

        Returns:
            tokenized samples where each sample is a list of token's id
        """
        ids_data = np.asarray([[np.argmax(token) for token in sample] for sample in data])

        return ids_data

    def _ids2onehot(self, data: Union[List[List[int]], Tuple[List[int]]], vocab_size: int) -> np.ndarray:
        """
        Convert token ids to one-hot representations in vocabulary of size vocab_size

        Args:
            data: list of tokenized samples where each sample is a list of token's id
            vocab_size: size of the vocabulary

        Returns:
            tokenized samples where each sample is a list of np.ndarrat of vocabulary size
        """
        onehot = np.eye(vocab_size)
        onehot_data = np.asarray([[onehot[token] for token in sample] for sample in data])

        return onehot_data

    def shutdown(self):
        self.sess.close()

    def reset(self):
        self.sess.close()
