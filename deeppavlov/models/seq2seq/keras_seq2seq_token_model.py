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
from overrides import overrides
from copy import deepcopy

from keras.layers import Dense, Input, Bidirectional
from keras.layers.recurrent import GRU, LSTM
from keras.layers.pooling import GlobalMaxPooling1D
from keras.models import Model
from keras.regularizers import l2

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.component import Component
from deeppavlov.models.classifiers.keras_classification_model import KerasClassificationModel
from deeppavlov.core.layers.keras_layers import multiplicative_self_attention
from deeppavlov.core.data.utils import zero_pad

log = get_logger(__name__)


@register("keras_seq2seq_token_model")
class KerasSeq2SeqTokenModel(KerasClassificationModel):
    """
    Class implements Keras model for seq2seq task on token-level

    Args:
        hidden_size: size of the hidden layer of encoder and decoder
        tgt_vocab_size: vocabulary size of target sequences
        tgt_pad_id: index of padding special token in target vocabulary
        tgt_bos_id: index of begin-of-sequence special token in target vocabulary
        tgt_eos_id: index of end-of-sequence special token in target vocabulary
        encoder_embedding_size: embedding size of encoder's embedder
        decoder_embedder: decoder's embedder component
        decoder_vocab: decoder's vocab component
        src_max_length: maximal token length of source sequence
        tgt_max_length: maximal token length of target sequence
        model_name: string name of particular method of this class that builds connection between encoder and decoder
        encoder_name: string name of particular method of this class that builds encoder model
        decoder_name: string name of particular method of this class that builds decoder model
        optimizer: string name of optimizer from keras.optimizers
        loss: string name of loss from keras.losses
        learning_rate learning rate for optimizer
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
                 tgt_vocab_size: int,
                 tgt_pad_id: int,
                 tgt_bos_id: int,
                 tgt_eos_id: int,
                 encoder_embedding_size: int,
                 decoder_embedder: Component,
                 decoder_vocab: Component,
                 maximal_response_length: int,
                 src_max_length: Optional[int] = None,
                 tgt_max_length: Optional[int] = None,
                 model_name: str = "encoder_decoder_model",
                 encoder_name: str = "gru_encoder_model",
                 decoder_name: str = "gru_decoder_model",
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
        decoder_embedding_size = kwargs.pop("decoder_embedding_size", decoder_embedder.dim)

        given_opt = {"hidden_size": hidden_size,
                     "tgt_vocab_size": tgt_vocab_size,
                     "tgt_pad_id": tgt_pad_id,
                     "tgt_bos_id": tgt_bos_id,
                     "tgt_eos_id": tgt_eos_id,
                     "encoder_embedding_size": encoder_embedding_size,
                     "decoder_embedding_size": decoder_embedding_size,
                     "maximal_response_length": maximal_response_length,
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

        self.decoder_embedder = decoder_embedder
        self.decoder_vocab = decoder_vocab

        self.encoder_model = None
        self.decoder_model = None
        self._encoder_emb_inp = None
        self._encoder_states = None
        self._decoder_emb_inp = None
        self._decoder_infer_emb_inp = None
        self._decoder_input_states = None
        self._train_decoder_states = None
        self._infer_decoder_states = None
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

    def texts2decoder_embeddings(self, sentences: List[List[str]],
                                 text_size: int, embedding_size: int,
                                 return_lengths: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Convert tokenized texts to vector representations using decoder_embedder \
        and padding up to text_size tokens

        Args:
            sentences: list of lists of tokens
            text_size: maximal number of tokens for padding
            embedding_size: embedding size
            return_lengths: whether to return lengths of each sample

        Returns:
            array of embedded texts, list of sentences lengths (optional)
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

    @overrides
    def pad_texts(self, sentences: Union[List[List[np.ndarray]], List[List[int]]],
                  text_size: int, embedding_size: int = None,
                  padding_token_id: int = 0,
                  return_lengths: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Cut and pad tokenized sequences (each sample is a list of indexed tokens or list of embedded tokens) \
        up to text_size tokens with zeros (or array of zeros of size embedding_size in case of embedded tokens)

        Args:
            sentences: list of lists of indexed or embedded tokens
            text_size: number of tokens to pad
            embedding_size: embedding size if sample is a list of embedded tokens
            padding_token_id: index of padding token in vocabulary
            return_lengths: whether to return lengths of each sample

        Returns:
            array of embedded texts, list of sentences lengths (optional)
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

    def encoder_decoder_model(self, **kwargs) -> Model:
        """
        Build keras models for training and infering

        Args:
            **kwargs: additional arguments

        Returns:
            keras model for training, also initializes encoder and decoder model separately for infering
        """
        self.encoder_method(**kwargs)

        self.decoder_method(**kwargs)

        encoder_decoder_model = Model(inputs=[self._encoder_emb_inp,
                                              self._decoder_emb_inp],
                                      outputs=self._train_decoder_outputs)

        self.encoder_model = Model(inputs=[self._encoder_emb_inp],
                                   outputs=self._encoder_states)

        self.decoder_model = Model(inputs=[self._decoder_infer_emb_inp] + self._decoder_input_states,
                                   outputs=[self._infer_decoder_outputs] + self._infer_decoder_states)

        return encoder_decoder_model

    @overrides
    def train_on_batch(self, x: Tuple[List[np.ndarray]], y: Tuple[List[int]], **kwargs) -> Union[float, List[float]]:
        """
        Train the self.model on the given batch using teacher forcing

        Args:
            x: list of tokenized embedded input samples
            y: list of output samples where each sample is a list of indices of tokens
            kwargs: additional arguments

        Returns:
            metrics values on the given batch
        """
        if self.opt["src_max_length"]:
            pad_emb_enc_inputs = self.pad_texts(x,
                                                text_size=self.opt["src_max_length"],
                                                embedding_size=self.opt["encoder_embedding_size"])
        else:
            pad_emb_enc_inputs = np.array(x)

        dec_inputs = [[self.opt["tgt_bos_id"]] +
                      list(sample) + [self.opt["tgt_eos_id"]]
                      for sample in y]  # (bs, ts + 2) of integers (tokens ids)
        text_dec_inputs = self.decoder_vocab(dec_inputs)

        dec_outputs = [list(sample) + [self.opt["tgt_eos_id"]] + [self.opt["tgt_pad_id"]]
                       for sample in y]  # (bs, ts + 2) of integers (tokens ids)

        if self.opt["tgt_max_length"]:
            pad_emb_dec_inputs = self.texts2decoder_embeddings(
                text_dec_inputs,
                text_size=self.opt["tgt_max_length"],
                embedding_size=self.opt["decoder_embedding_size"])
            pad_dec_outputs = self.pad_texts(dec_outputs,
                                             text_size=self.opt["tgt_max_length"],
                                             embedding_size=None,
                                             padding_token_id=self.opt["tgt_pad_id"])
        else:
            pad_emb_dec_inputs = self.decoder_embedder(text_dec_inputs)
            pad_dec_outputs = zero_pad(dec_outputs, padding=self.opt["tgt_pad_id"], dtype=np.int32)

        pad_onehot_dec_outputs = self._ids2onehot(pad_dec_outputs, vocab_size=self.opt["tgt_vocab_size"])

        metrics_values = self.model.train_on_batch([pad_emb_enc_inputs,
                                                    pad_emb_dec_inputs],
                                                   pad_onehot_dec_outputs)
        return metrics_values

    @overrides
    def infer_on_batch(self, x: List[List[np.ndarray]], **kwargs) -> List[List[int]]:
        """
        Infer self.encoder_model and self.decoder_model on the given data (no teacher forcing)

        Args:
            x: list of tokenized embedded input samples
            **kwargs: additional arguments

        Returns:
            list of decoder predictions where each prediction is a list of indices of tokens
        """
        if self.opt["src_max_length"]:
            pad_emb_enc_inputs = self.pad_texts(x,
                                                text_size=self.opt["src_max_length"],
                                                embedding_size=self.opt["encoder_embedding_size"])
        else:
            pad_emb_enc_inputs = np.array(x)

        encoder_states = self.encoder_model.predict([pad_emb_enc_inputs])

        predicted_batch = []
        for i in range(len(x)):  # batch size
            predicted_sample = []

            current_token = self.decoder_embedder([[self.decoder_vocab[self.opt["tgt_bos_id"]]]])[0][0]  # (300,)
            end_of_sequence = False

            if len(encoder_states.shape) > 2:
                states = [enc_state[i].reshape((1, -1)) for enc_state in encoder_states]
            else:
                states = [encoder_states[i].reshape((1, -1))]  #???

            while not end_of_sequence:
                out = self.decoder_model.predict([np.array([[current_token]])] + states)
                token_probas = out[0]
                states = out[1:]
                current_token_id = self._probas2ids(token_probas)[0][0]
                current_token = self.decoder_embedder(self.decoder_vocab([[current_token_id]]))[0][0]
                if (current_token_id == self.opt["tgt_eos_id"] or
                        current_token_id == self.opt["tgt_pad_id"] or
                        len(predicted_sample) == self.opt["maximal_response_length"]):
                    end_of_sequence = True
                else:
                    predicted_sample.append(current_token_id)

            predicted_batch.append(predicted_sample)

        return predicted_batch

    @overrides
    def __call__(self, x: List[List[np.ndarray]], **kwargs) -> np.ndarray:
        """
        Infer self.encoder_model and self.decoder_model on the given data (no teacher forcing)

        Args:
            x: list of tokenized embedded input samples
            **kwargs: additional arguments

        Returns:
            array of decoder predictions where each prediction is a list of indices of tokens
        """
        predictions = np.array(self.infer_on_batch(x))
        return predictions

    @staticmethod
    def _probas2ids(data: List[List[np.ndarray]]) -> np.ndarray:
        """
        Convert vectors of probabilities distribution of tokens in the vocabulary \
        or one-hot token representations in the vocabulary  \
        to corresponding token ids

        Args:
            data: list of tokenized samples where each sample is a list of np.ndarray of vocabulary size

        Returns:
            tokenized samples where each sample is a list of token's ids
        """
        ids_data = np.asarray([[np.argmax(token) for token in sample] for sample in data])

        return ids_data

    @staticmethod
    def _ids2onehot(data: Union[List[List[int]], Tuple[List[int]]], vocab_size: int) -> np.ndarray:
        """
        Convert token ids to one-hot representations in vocabulary of size vocab_size

        Args:
            data: list of tokenized samples where each sample is a list of token's ids
            vocab_size: size of the vocabulary

        Returns:
            tokenized samples where each sample is a list of np.ndarrays of vocabulary size
        """
        onehot = np.eye(vocab_size)
        onehot_data = np.asarray([[onehot[token] for token in sample] for sample in data])

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

    def gru_encoder_model(self,
                          hidden_size: int,
                          encoder_coef_reg_lstm: float,
                          encoder_dropout_rate: float,
                          encoder_rec_dropout_rate: float,
                          **kwargs) -> None:
        """
        Initialize encoder layers for GRU encoder

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

        _encoder_outputs = GRU(
            hidden_size,
            activation='tanh',
            return_sequences=True,  # for extracting exactly the last hidden layer
            kernel_regularizer=l2(encoder_coef_reg_lstm),
            dropout=encoder_dropout_rate,
            recurrent_dropout=encoder_rec_dropout_rate,
            name="encoder_gru")(self._encoder_emb_inp)
        self._encoder_states = [GlobalMaxPooling1D()(_encoder_outputs)]

        return None

    def gru_decoder_model(self,
                          hidden_size: int,
                          decoder_coef_reg_lstm: float,
                          decoder_dropout_rate: float,
                          decoder_rec_dropout_rate: float,
                          **kwargs) -> None:
        """
        Initialize decoder layers for GRU decoder

        Args:
            hidden_size: size of the hidden layer of encoder and decoder
            decoder_coef_reg_lstm: coefficient for L2 kernel regularizer of decoder LSTM layer
            decoder_dropout_rate: dropout rate for decoder LSTM layer
            decoder_rec_dropout_rate: recurrent dropout rate for decoder LSTM layer

        Returns:
            None
        """
        self._decoder_emb_inp = Input(shape=(self.opt["tgt_max_length"], self.opt["decoder_embedding_size"]))
        self._decoder_infer_emb_inp = Input(shape=(1, self.opt["decoder_embedding_size"]))

        self._decoder_input_states = [Input(shape=(hidden_size,))]

        decoder_gru = GRU(
            hidden_size,
            activation='tanh',
            return_sequences=True,
            kernel_regularizer=l2(decoder_coef_reg_lstm),
            dropout=decoder_dropout_rate,
            recurrent_dropout=decoder_rec_dropout_rate,
            name="decoder_gru")
        _train_decoder_outputs = decoder_gru(
            self._decoder_emb_inp,
            initial_state=self._encoder_states[0])
        self._train_decoder_states = [GlobalMaxPooling1D()(_train_decoder_outputs)]

        _infer_decoder_outputs = decoder_gru(
            self._decoder_infer_emb_inp,
            initial_state=self._decoder_input_states[0])
        self._infer_decoder_states = [GlobalMaxPooling1D()(_infer_decoder_outputs)]

        decoder_dense = Dense(self.opt["tgt_vocab_size"], name="dense_gru", activation="softmax")
        self._train_decoder_outputs = decoder_dense(_train_decoder_outputs)
        self._infer_decoder_outputs = decoder_dense(_infer_decoder_outputs)

        return None

    def bilstm_encoder_model(self,
                             hidden_size: int,
                             encoder_coef_reg_lstm: float,
                             encoder_dropout_rate: float,
                             encoder_rec_dropout_rate: float,
                             **kwargs) -> None:
        """
        Initialize encoder layers for BiLSTM encoder

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

        _encoder_outputs, forward_state_h, forward_state_c, backward_state_h, backward_state_c = Bidirectional(LSTM(
            hidden_size,
            activation='tanh',
            return_sequences=True,  # for extracting exactly the last hidden layer
            return_state=True,
            kernel_regularizer=l2(encoder_coef_reg_lstm),
            dropout=encoder_dropout_rate,
            recurrent_dropout=encoder_rec_dropout_rate,
            name="encoder_gru"))(self._encoder_emb_inp)

        self._encoder_states = [forward_state_h, forward_state_c, backward_state_h, backward_state_c]

        return None

    def bilstm_decoder_model(self,
                             hidden_size: int,
                             decoder_coef_reg_lstm: float,
                             decoder_dropout_rate: float,
                             decoder_rec_dropout_rate: float,
                             **kwargs) -> None:
        """
        Initialize decoder layers for BiLSTM decoder

        Args:
            hidden_size: size of the hidden layer of encoder and decoder
            decoder_coef_reg_lstm: coefficient for L2 kernel regularizer of decoder LSTM layer
            decoder_dropout_rate: dropout rate for decoder LSTM layer
            decoder_rec_dropout_rate: recurrent dropout rate for decoder LSTM layer

        Returns:
            None
        """
        self._decoder_emb_inp = Input(shape=(self.opt["tgt_max_length"], self.opt["decoder_embedding_size"]))
        self._decoder_infer_emb_inp = Input(shape=(1, self.opt["decoder_embedding_size"]))

        self._decoder_input_states = [Input(shape=(hidden_size,)),
                                      Input(shape=(hidden_size,)),
                                      Input(shape=(hidden_size,)),
                                      Input(shape=(hidden_size,))]

        decoder_gru = Bidirectional(LSTM(
            hidden_size,
            activation='tanh',
            return_sequences=True,
            return_state=True,
            kernel_regularizer=l2(decoder_coef_reg_lstm),
            dropout=decoder_dropout_rate,
            recurrent_dropout=decoder_rec_dropout_rate,
            name="decoder_gru"))

        _train_decoder_outputs, forward_state_h, forward_state_c, backward_state_h, backward_state_c = decoder_gru(
            self._decoder_emb_inp,
            initial_state=self._encoder_states)
        self._train_decoder_states = [forward_state_h, forward_state_c, backward_state_h, backward_state_c]

        _infer_decoder_outputs, forward_state_h, forward_state_c, backward_state_h, backward_state_c = decoder_gru(
            self._decoder_infer_emb_inp,
            initial_state=self._decoder_input_states)
        self._infer_decoder_states = [forward_state_h, forward_state_c, backward_state_h, backward_state_c]

        decoder_dense = Dense(self.opt["tgt_vocab_size"], name="dense_gru", activation="softmax")
        self._train_decoder_outputs = decoder_dense(_train_decoder_outputs)
        self._infer_decoder_outputs = decoder_dense(_infer_decoder_outputs)

        return None
