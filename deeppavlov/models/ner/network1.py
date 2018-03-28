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
from tensorflow.contrib.layers import xavier_initializer, xavier_initializer_conv2d

from deeppavlov.core.common.log import get_logger
from deeppavlov.core.layers.tf_layers import character_embedding_network
from deeppavlov.core.layers.tf_layers import embedding_layer
from deeppavlov.core.layers.tf_layers import stacked_cnn
from deeppavlov.core.layers.tf_layers import stacked_highway_cnn
from deeppavlov.core.layers.tf_layers import stacked_bi_rnn
from deeppavlov.models.ner.evaluation import precision_recall_f1

import pickle

SEED = 42
MODEL_FILE_NAME = 'ner_model'

log = get_logger(__name__)


class NerNetwork:
    def __init__(self,                
                 dicts_file,                 
                 word_dim,
                 word_hidden_size,
                 char_dim,
                 nb_filters_1,
                 nb_filters_2,
                 cap_dim,
                 cap_hidden_size,
                 drop_out,
                 pretrained_emb=None,
                 sess=None):
        # load dictionaries
        dicts = pickle.load(open(dicts_file, mode="rb"))
        self.word2id = dicts["word2id"]
        self.id2word = dicts["id2word"]
        self.char2id = dicts["char2id"]
        self.id2char = dicts["id2char"]
        self.tag2id = dicts["tag2id"]
        self.id2tag = dicts["id2tag"]

        self.word_vocab_size = len(self.word2id)
        self.char_vocab_size = len(self.char2id)
        self.tag_vocab_size = len(self.tag2id)

        self.word_dim = word_dim
        self.word_hidden_size = word_hidden_size
        self.char_dim = char_dim
        self.nb_filters_1 = nb_filters_1
        self.nb_filters_2 = nb_filters_2
        self.cap_dim = cap_dim
        self.cap_hidden_size = cap_hidden_size
        self.drop_out = drop_out

        self.lower = 1
        self.zeros = 0

        self.tf_word_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="word_ids")
        self.tf_sentence_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name="sentence_lengths")
        self.tf_labels = tf.placeholder(dtype=tf.int32, shape=[None, None], name="labels")

        self.tf_dropout = tf.placeholder(dtype=tf.float32, shape=[], name="drop_out")
        self.tf_learning_rate= tf.placeholder(dtype=tf.float32, shape=[], name="learning_rate")

        self.tf_char_ids = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name="char_ids")
        self.tf_word_lengths = tf.placeholder(dtype=tf.int32, shape=[None, None], name="word_lengths")

        self.tf_cap_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="cap_ids")

        # embedd word input
        with tf.variable_scope("word_input"):
            if pretrained_emb is not None:
                self.pretrained_emb = self.read_glove_emb(pretrained_emb, self.word2id,
                    self.word_vocab_size, self.word_dim, lower=True, zeros=False)
            else:
                self.pretrained_emb = np.zeros(shape=(len(self.word2id), self.params.word_dim))

            tf_word_embeddings = tf.Variable(self.pretrained_emb, dtype=tf.float32,
                trainable=True, name="word_embedding")
            
            embedded_words = tf.nn.embedding_lookup(tf_word_embeddings, self.tf_word_ids, name="embedded_words")
            self.input = embedded_words

        # extract character features using CNN
        with tf.variable_scope("char_cnn"):
            tf_char_embeddings = tf.get_variable(name="char_embeddings",
                                                 dtype=tf.float32,
                                                 shape=[self.char_vocab_size, self.char_dim],
                                                 trainable=True)
            embedded_chars = tf.nn.embedding_lookup(tf_char_embeddings,
                                                        self.tf_char_ids,
                                                        name="embedded_chars")

            conv1 = tf.layers.conv2d(inputs=embedded_chars,
                                        filters=self.nb_filters_1,
                                        kernel_size=(1, 3),
                                        strides=(1, 1),
                                        padding="same",
                                        name="conv1",
                                        kernel_initializer=xavier_initializer_conv2d())
            conv2 = tf.layers.conv2d(inputs=conv1,
                                        filters=self.nb_filters_2,
                                        kernel_size=(1, 3),
                                        strides=(1, 1),
                                        padding="same",
                                        name="conv2",
                                        kernel_initializer=xavier_initializer_conv2d())
            char_cnn = tf.reduce_max(conv2, axis=2)

            self.input = tf.concat([self.input, char_cnn], axis=-1)
            self.input = tf.nn.dropout(self.input, self.tf_dropout)

        # extract cap. features using Bi-LSTM
        with tf.variable_scope("cap_bilstm"):
            cap_embeddings = tf.get_variable(name="cap_embeddings", dtype=tf.float32, shape=[5, self.cap_dim], trainable=True)
            ebedded_caps = tf.nn.embedding_lookup(cap_embeddings, self.tf_cap_ids, name="ebedded_caps")
            cap_cell_fw = tf.contrib.rnn.LSTMCell(self.cap_hidden_size)
            cap_cell_bw = tf.contrib.rnn.LSTMCell(self.cap_hidden_size)
            (cap_output_fw, cap_output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cap_cell_fw,
                                                                                cap_cell_bw, ebedded_caps,
                                                                                sequence_length=self.tf_sentence_lengths,
                                                                                dtype=tf.float32)
            cap_output = tf.concat([cap_output_fw, cap_output_bw], axis=-1)
            self.input = tf.concat([self.input, cap_output], axis=-1)
            self.input = tf.nn.dropout(self.input, self.tf_dropout)

        # extract contextual word features using Bi-LSTM
        with tf.variable_scope("bi_lstm_words"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.word_hidden_size)
            cell_bw = tf.contrib.rnn.LSTMCell(self.word_hidden_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.input,
                                                                        sequence_length=self.tf_sentence_lengths,
                                                                        dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
        
            ntime_steps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.word_hidden_size])
            layer1 = tf.nn.dropout(tf.layers.dense(inputs=output, units=256, activation=tf.tanh, kernel_initializer=xavier_initializer()), self.tf_dropout)
            pred = tf.layers.dense(inputs=layer1, units=len(self.tag2id), activation=None, kernel_initializer=xavier_initializer())
            self.logits = tf.reshape(pred, [-1, ntime_steps, len(self.tag2id)])

            # compute loss value using crf
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.logits,
                                                                                       self.tf_labels,
                                                                                       self.tf_sentence_lengths)
            
        self.tf_loss = tf.reduce_mean(-log_likelihood)
        self.tf_train_op = tf.train.AdamOptimizer(learning_rate=self.tf_learning_rate).minimize(self.tf_loss)

        # Initialize session
        if sess is None:
            sess = tf.Session()        

        self._sess = sess
        sess.run(tf.global_variables_initializer())

    def read_glove_emb(self, emb_path, word_to_id, vocab_size, word_dim, lower, zeros):
        print ("Pre-trained word embedding is being loaded.")
        word_vectors = dict()
        loaded_words = 0
        def l(x): return x.lower() if lower else x
        def z(s): return re.sub('\d', '0', s) if zeros else s

        counter = 0
        with open (emb_path, encoding="utf8") as f:
            next(f)
            for line in f:          
                items = line.split()
                token = items[:-word_dim]

                # skip the words containing more than one word
                if len (token) != 1:
                    continue
                word = l(z(token[0]))
                if word not in word_vectors:
                    word_vectors[word] = np.array([float(num_str) for num_str in items[1:]])
                        
        pretrained_words = np.zeros(shape=(vocab_size, word_dim))
        for word in word_to_id:
            if word in word_vectors:
                pretrained_words[word_to_id[word]] = word_vectors[word]
                loaded_words += 1

        print ("There are {:} words in the vocabulary and {:} words were loaded from pre-trained embedding".format (len (word_to_id),
                                                                                                          loaded_words))
        return pretrained_words    

    def tokens_batch_to_numpy_batch(self, batch_x, batch_y=None):
        def l(x): return x.lower() if self.lower == 1 else x
        def z(s): return helper.zeros(s) if self.zeros == 1 else s
        def cap_feature(s):
            if s.upper() == s:
                return 1
            elif s.lower() == s:
                return 2
            elif (s[0].upper() == s[0]) and (s[1:].lower() == s[1:]):
                return 3
            else:
                return 4

        word = [[l(z(x)) for x in s] for s in batch_x]

        # index data
        indexed_word = [[self.word2id[w] if w in self.word2id else self.word2id["<UNK>"] for w in s] for s in word]
        # indexed_data = {"indexed_word": indexed_word}

        if batch_y is not None:
            indexed_tag= [[self.tag2id[t] for t in s] for s in batch_y]

        indexed_char = [[[self.char2id[c] if c in self.char2id else self.char2id["<UNK>"] for c in z(w)] for w in s] for s in batch_x]
        # indexed_data["indexed_char"] = indexed_char

        indexed_cap = [[cap_feature(w) for w in s] for s in batch_x]
        # indexed_data["indexed_cap"] = indexed_cap

        # pad word and tag
        real_sentence_lengths = [len(sent) for sent in indexed_word]
        max_len_sentences = max(real_sentence_lengths)
        padded_word = [np.lib.pad(sent, (0, max_len_sentences - len(sent)), 'constant',
            constant_values=(self.word2id["<PAD>"], self.word2id["<PAD>"])) for sent in indexed_word]

        batch = {"batch_word": indexed_word, "padded_word": padded_word,
                 "real_sentence_lengths": real_sentence_lengths}

        if batch_y is not None:
            padded_tag = [np.lib.pad(sent, (0, max_len_sentences - len(sent)), 'constant',
                constant_values=(self.tag2id["<PAD>"], self.tag2id["<PAD>"])) for sent in indexed_tag]
            batch["batch_tag"] = indexed_tag,
            batch["padded_tag"] = padded_tag

        # pad chars
        max_len_of_sentence = max([len(sentence) for sentence in indexed_char])
        max_len_of_word = max([max([len(word) for word in sentence]) for sentence in indexed_char])

        padding_word = np.full(max_len_of_word, self.char2id["<PAD>"])
        padded_char = []

        lengths_of_word = []

        for sentence in indexed_char:
            padded_sentence = []
            length_of_word_in_sentence = []

            for word in sentence:
                length_of_word_in_sentence.append(len(word))
                padded_sentence.append(np.lib.pad(word, (0, max_len_of_word - len(word)), 'constant',
                                                  constant_values=(self.char2id["<PAD>"], self.char2id["<PAD>"])))

            for i in range(max_len_of_sentence - len(padded_sentence)):
                padded_sentence.append(padding_word)
                length_of_word_in_sentence.append(0)

            padded_char.append(padded_sentence)
            lengths_of_word.append(length_of_word_in_sentence)

        lengths_of_word = np.array(lengths_of_word)

        batch["padded_char"] = padded_char
        batch["lengths_of_word"] = lengths_of_word

        # pad cap
        padded_cap = [np.lib.pad(x, (0, max_len_sentences - len(x)), 'constant',
                             constant_values=(0, 0)) for x in indexed_cap]
        batch["padded_cap"] = padded_cap

        return batch

    
    def train_on_batch(self, batch_x, batch_y, learning_rate=1e-3, dropout_rate=0.5):
        batch = self.tokens_batch_to_numpy_batch(batch_x, batch_y)
        feed_dict = {self.tf_word_ids: batch["padded_word"],
                        self.tf_sentence_lengths: batch["real_sentence_lengths"],
                        self.tf_char_ids: batch["padded_char"],
                        self.tf_word_lengths: batch["lengths_of_word"],
                        self.tf_cap_ids: batch["padded_cap"],
                        self.tf_labels: batch["padded_tag"],     
                        self.tf_learning_rate: learning_rate,
                        self.tf_dropout: dropout_rate}
        loss, _ = self._sess.run([self.tf_loss, self.tf_train_op], feed_dict=feed_dict)
        return loss

    def predict_on_batch(self, batch_x):
        batch = self.tokens_batch_to_numpy_batch(batch_x)
        feed_dict = {self.tf_word_ids: batch["padded_word"],
                        self.tf_sentence_lengths: batch["real_sentence_lengths"],
                        self.tf_char_ids: batch["padded_char"],
                        self.tf_word_lengths: batch["lengths_of_word"],
                        self.tf_cap_ids: batch["padded_cap"],
                        # self.tf_learning_rate: learning_rate,
                        self.tf_dropout: 1.0}
        _logits, _transition_params = self._sess.run([self.logits, self.transition_params], feed_dict=feed_dict)

        y_pred = []
        # iterate over the sentences
        for _logit, sequence_length in zip(_logits, batch["real_sentence_lengths"]):
            # keep only the valid time steps
            _logit = _logit[:sequence_length]
            viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode (_logit, _transition_params)
            y_pred += [viterbi_sequence]

        y_pred_tag = [[self.id2tag[t] for t in sent] for sent in y_pred]

        return y_pred_tag

    def shutdown(self):
        self._sess.close()

    def save(self, model_file_path):
        """
        Save model to model_file_path
        """
        saver = tf.train.Saver()
        saver.save(self._sess, str(model_file_path))

    def load(self, model_file_path):
        """
        Load model from the model_file_path
        """
        saver = tf.train.Saver()
        saver.restore(self._sess, str(model_file_path))

    @staticmethod
    def get_trainable_variables(trainable_scope_names=None):
        vars = tf.trainable_variables()
        if trainable_scope_names is not None:
            vars_to_train = []
            for scope_name in trainable_scope_names:
                for var in vars:
                    if var.name.startswith(scope_name):
                        vars_to_train.append(var)
            return vars_to_train
        else:
            return vars

    def get_train_op(self, loss, learning_rate, learnable_scopes=None, optimizer=None):
        """ Get train operation for given loss

        Args:
            loss: loss, tf tensor or scalar
            learning_rate: scalar or placeholder
            learnable_scopes: which scopes are trainable (None for all)
            optimizer: instance of tf.train.Optimizer, default Adam

        Returns:
            train_op
        """
        variables = self.get_trainable_variables(learnable_scopes)
        if optimizer is None:
            optimizer = tf.train.AdamOptimizer

        # For batch norm it is necessary to update running averages
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = optimizer(learning_rate).minimize(loss, var_list=variables)
        return train_op