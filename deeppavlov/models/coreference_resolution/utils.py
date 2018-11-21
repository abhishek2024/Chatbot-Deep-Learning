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

import os
import uuid
import time
import operator
# import fastText
from fasttext import load_model
import numpy as np
import collections
import tensorflow as tf
import shutil

from tqdm import tqdm
from os.path import join
from shutil import copytree
from collections import defaultdict
from sklearn.model_selection import ShuffleSplit
from deeppavlov.core.data.utils import download_decompress


seed = 5
np.random.seed(seed)


def dict2conll(data, predict):
    """
    Creates conll document, and write there predicted conll string.
    Args:
        data: dict from agent with conll string
        predict: string with address and name of file

    Returns: Nothing

    """
    with open(predict, 'w') as CoNLL:
        CoNLL.write(data['conll_str'])
    return None


def normalize(v):
    """Normalize input tensor"""
    norm = np.linalg.norm(v)
    if norm > 0:
        return v / norm
    else:
        return v


def flatten(l):
    """Expands list"""
    return [item for sublist in l for item in sublist]


def load_char_dict(char_vocab_path):
    """Load char dict from file"""
    vocab = [u"<unk>"]
    with open(char_vocab_path) as f:
        vocab.extend(c.strip() for c in f.readlines())
    char_dict = collections.defaultdict(int)
    char_dict.update({c: i for i, c in enumerate(sorted(set(vocab)))})
    return char_dict


def load_embedding_dict(embedding_path, embedding_size, embedding_format):
    """
    Load emb dict from file, or load pre trained binary fasttext model.
    Args:
        embedding_path: path to the vec file, or binary model
        embedding_size: int, embedding_size
        embedding_format: 'bin' or 'vec'

    Returns: Embeddings dict, or fasttext pre trained model

    """
    print("Loading word embeddings from {}...".format(embedding_path))

    if embedding_format == 'vec':
        default_embedding = np.zeros(embedding_size)
        embedding_dict = collections.defaultdict(lambda: default_embedding)
        skip_first = embedding_format == "vec"
        with open(embedding_path) as f:
            for i, line in enumerate(f.readlines()):
                if skip_first and i == 0:
                    continue
                splits = line.split()
                assert len(splits) == embedding_size + 1
                word = splits[0]
                embedding = np.array([float(s) for s in splits[1:]])
                embedding_dict[word] = embedding
    elif embedding_format == 'bin':
        embedding_dict = load_model(embedding_path)
        # embedding_dict = fastText.load_model(embedding_path)
    else:
        raise ValueError('Not supported embeddings format {}'.format(embedding_format))
    print("Done loading word embeddings.")
    return embedding_dict


def maybe_divide(x, y):
    return 0 if y == 0 else x / float(y)


def projection(inputs, output_size, initializer=None):
    """ Returns outputs of fully-connected network """
    return ffnn(inputs, 0, -1, output_size, dropout=None, output_weights_initializer=initializer)


def shape(x, dim):
    """ Returns shape of tensor """
    return x.get_shape()[dim].value or tf.shape(x)[dim]


#  Networks
def ffnn(inputs, num_hidden_layers, hidden_size, output_size, dropout,
         output_weights_initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float64)):
    """
    Creates fully_connected network graph with needed parameters.
    Args:
        inputs: shape of input tensor, rank of input >3 not supported
        num_hidden_layers: int32
        hidden_size: int32
        output_size: int32
        dropout: int32, dropout value
        output_weights_initializer: name of initializers or None

    Returns: network output, [output_size]

    """

    # inputs = tf.cast(inputs, tf.float32)
    if len(inputs.get_shape()) > 2:
        current_inputs = tf.reshape(inputs, [-1, shape(inputs, -1)])
    else:
        current_inputs = inputs

    for i in range(num_hidden_layers):
        hidden_weights = tf.get_variable("hidden_weights_{}".format(i), [shape(current_inputs, 1), hidden_size],
                                         dtype=tf.float64)
        hidden_bias = tf.get_variable("hidden_bias_{}".format(i), [hidden_size], dtype=tf.float64)

        current_outputs = tf.nn.relu(tf.matmul(current_inputs, hidden_weights) + hidden_bias)

        if dropout is not None:
            current_outputs = tf.nn.dropout(current_outputs, dropout)
        current_inputs = current_outputs

    output_weights = tf.get_variable("output_weights", [shape(current_inputs, 1), output_size],
                                     dtype=tf.float64)
    output_bias = tf.get_variable("output_bias", [output_size], dtype=tf.float64)
    outputs = tf.matmul(current_inputs, output_weights) + output_bias
    # outputs = tf.cast(outputs, tf.float64)
    if len(inputs.get_shape()) == 3:
        outputs = tf.reshape(outputs, [shape(inputs, 0), shape(inputs, 1), output_size])
    elif len(inputs.get_shape()) > 3:
        raise ValueError("FFNN with rank {} not supported".format(len(inputs.get_shape())))
    return outputs


def cnn(inputs, filter_sizes, num_filters):
    """
    Creates convolutional network graph with needed parameters.
    Args:
        inputs: shape of input tensor
        filter_sizes: list of shapes of filters
        num_filters: amount of filters

    Returns: network output, [num_words, num_filters * len(filter_sizes)]

    """
    num_words = shape(inputs, 0)
    num_chars = shape(inputs, 1)
    input_size = shape(inputs, 2)
    outputs = []

    # TODO: del the tf.cast(float32) when https://github.com/tensorflow/tensorflow/pull/12943 will be done
    for i, filter_size in enumerate(filter_sizes):
        with tf.variable_scope("conv_{}".format(i)):
            w = tf.get_variable("w", [filter_size, input_size, num_filters])
            b = tf.get_variable("b", [num_filters], dtype=tf.float64)
        conv = tf.nn.conv1d(tf.cast(inputs, tf.float32), w, stride=1, padding="VALID")  # [num_words, num_chars - filter_size, num_filters]
        h = tf.nn.relu(tf.nn.bias_add(tf.cast(conv, tf.float64), b))  # [num_words, num_chars - filter_size, num_filters]
        pooled = tf.reduce_max(h, 1)  # [num_words, num_filters]
        outputs.append(pooled)
    return tf.concat(outputs, 1)  # [num_words, num_filters * len(filter_sizes)]


class CustomLSTMCell(tf.contrib.rnn.RNNCell):
    """Bi-LSTM"""

    def __init__(self, num_units, batch_size, dropout):
        """Initialize graph"""
        self._num_units = num_units
        self._dropout = dropout
        self._dropout_mask = tf.nn.dropout(tf.ones([batch_size, self.output_size], dtype=tf.float64), dropout)
        self._initializer = self._block_orthonormal_initializer([self.output_size] * 3)
        initial_cell_state = tf.get_variable("lstm_initial_cell_state", [1, self.output_size], dtype=tf.float64)
        initial_hidden_state = tf.get_variable("lstm_initial_hidden_state", [1, self.output_size], dtype=tf.float64)
        self._initial_state = tf.contrib.rnn.LSTMStateTuple(initial_cell_state, initial_hidden_state)

    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self.output_size, self.output_size)

    @property
    def output_size(self):
        return self._num_units

    @property
    def initial_state(self):
        return self._initial_state

    def preprocess_input(self, inputs):
        return projection(inputs, 3 * self.output_size)

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # "CustomLSTMCell"
            c, h = state
            h *= self._dropout_mask
            projected_h = projection(h, 3 * self.output_size, initializer=self._initializer)
            concat = inputs + projected_h
            i, j, o = tf.split(concat, num_or_size_splits=3, axis=1)
            i = tf.sigmoid(i)
            new_c = (1 - i) * c + i * tf.tanh(j)
            new_h = tf.tanh(new_c) * tf.sigmoid(o)
            new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
            return new_h, new_state

    @staticmethod
    def _orthonormal_initializer(scale=1.0):
        def _initializer(shape, dtype=tf.float64, partition_info=None):
            M1 = np.random.randn(shape[0], shape[0]).astype(np.float64)
            M2 = np.random.randn(shape[1], shape[1]).astype(np.float64)
            Q1, R1 = np.linalg.qr(M1)
            Q2, R2 = np.linalg.qr(M2)
            Q1 = Q1 * np.sign(np.diag(R1))
            Q2 = Q2 * np.sign(np.diag(R2))
            n_min = min(shape[0], shape[1])
            params = np.dot(Q1[:, :n_min], Q2[:n_min, :]) * scale
            return params

        return _initializer

    def _block_orthonormal_initializer(self, output_sizes):
        def _initializer(shape, dtype=np.float64, partition_info=None):
            assert len(shape) == 2
            assert sum(output_sizes) == shape[1]
            initializer = self._orthonormal_initializer()
            params = np.concatenate([initializer([shape[0], o], dtype, partition_info) for o in output_sizes], 1)
            return params

        return _initializer


class DocumentState(object):
    """
    Class that verifies the conll document and creates on its basis a new dictionary.
    """

    def __init__(self):
        self.doc_key = None
        self.text = []
        self.text_speakers = []
        self.speakers = []
        self.sentences = []
        self.clusters = collections.defaultdict(list)
        self.stacks = collections.defaultdict(list)

    def assert_empty(self):
        assert self.doc_key is None, 'self.doc_key is None'
        assert len(self.text) == 0, 'len(self.text) == 0'
        assert len(self.text_speakers) == 0, 'len(self.text_speakers)'
        assert len(self.sentences) == 0, 'len(self.sentences) == 0'
        assert len(self.speakers) == 0, 'len(self.speakers) == 0'
        assert len(self.clusters) == 0, 'len(self.clusters) == 0'
        assert len(self.stacks) == 0, 'len(self.stacks) == 0'

    def assert_finalizable(self):
        assert self.doc_key is not None, 'self.doc_key is not None finalizable'
        assert len(self.text) == 0, 'len(self.text) == 0_finalizable'
        assert len(self.text_speakers) == 0, 'len(self.text_speakers) == 0_finalizable'
        assert len(self.sentences) > 0, 'len(self.sentences) > 0_finalizable'
        assert len(self.speakers) > 0, 'len(self.speakers) > 0_finalizable'
        assert all(
            len(s) == 0 for s in self.stacks.values()), 'all(len(s) == 0 for s in self.stacks.values())_finalizable'

    def finalize(self):
        merged_clusters = []
        for c1 in self.clusters.values():
            existing = None
            for m in c1:
                for c2 in merged_clusters:
                    if m in c2:
                        existing = c2
                        break
                if existing is not None:
                    break
            if existing is not None:
                # print("Merging clusters (shouldn't happen very often.)")
                existing.update(c1)
            else:
                merged_clusters.append(set(c1))
        merged_clusters = [list(c) for c in merged_clusters]
        all_mentions = flatten(merged_clusters)
        # In folder test one file have repeting mentions, it call below assert, everething else works fine

        return {"doc_key": self.doc_key,
                "sentences": self.sentences,
                "speakers": self.speakers,
                "clusters": merged_clusters}


def normalize_word(word):
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word


def handle_line(line, document_state):
    """
    Updates the state of the document_state class in a read string of conll document.
    Args:
        line: line from conll document
        document_state: analyzing class

    Returns: document_state, or Nothing

    """
    if line.startswith("#begin"):
        document_state.assert_empty()
        row = line.split()
        #    document_state.doc_key = 'bc'+'{0}_{1}'.format(row[2][1:-2],row[-1])
        return None
    elif line.startswith("#end document"):
        document_state.assert_finalizable()
        return document_state.finalize()
    else:
        row = line.split('\t')
        if len(row) == 1:
            document_state.sentences.append(tuple(document_state.text))
            del document_state.text[:]
            document_state.speakers.append(tuple(document_state.text_speakers))
            del document_state.text_speakers[:]
            return None
        assert len(row) >= 12, 'len < 12'

        word = normalize_word(row[3])
        coref = row[-1]
        if document_state.doc_key is None:
            document_state.doc_key = 'bc' + '{0}_{1}'.format(row[0], row[1])

        speaker = row[8]

        word_index = len(document_state.text) + sum(len(s) for s in document_state.sentences)
        document_state.text.append(word)
        document_state.text_speakers.append(speaker)

        if coref == "-":
            return None

        for segment in coref.split("|"):
            if segment[0] == "(":
                if segment[-1] == ")":
                    cluster_id = segment[1:-1]  # here was int
                    document_state.clusters[cluster_id].append((word_index, word_index))
                else:
                    cluster_id = segment[1:]  # here was int
                    document_state.stacks[cluster_id].append(word_index)
            else:
                cluster_id = segment[:-1]  # here was int
                start = document_state.stacks[cluster_id].pop()
                document_state.clusters[cluster_id].append((start, word_index))
        return None


def conll2modeldata(conll_str):
    """
    Converts the document into a dictionary, with the required format for the model.
    Args:
        conll_str: dict with conll string

    Returns: dict like:

    {
      "clusters": [[[1024,1024],[1024,1025]],[[876,876], [767,765], [541,544]]],
      "doc_key": "nw",
      "sentences": [["This", "is", "the", "first", "sentence", "."], ["This", "is", "the", "second", "."]],
      "speakers": [["spk1", "spk1", "spk1", "spk1", "spk1", "spk1"], ["spk2", "spk2", "spk2", "spk2", "spk2"]]
    }

    """
    model_file = None
    document_state = DocumentState()
    line_list = conll_str.split('\n')
    for line in line_list:
        document = handle_line(line, document_state)
        if document is not None:
            model_file = document
    return model_file


def output_conll(input_file, predictions):
    """
    Gets the string with conll file, and write there new coreference clusters from predictions.

    Args:
        input_file: dict with conll string
        predictions: dict new clusters

    Returns: modified string with conll file

    """
    prediction_map = {}
    # input_file = input_file['conll_str']
    for doc_key, clusters in predictions.items():
        start_map = collections.defaultdict(list)
        end_map = collections.defaultdict(list)
        word_map = collections.defaultdict(list)
        for cluster_id, mentions in enumerate(clusters):
            for start, end in mentions:
                if start == end:
                    word_map[start].append(cluster_id)
                else:
                    start_map[start].append((cluster_id, end))
                    end_map[end].append((cluster_id, start))
        for k, v in start_map.items():
            start_map[k] = [cluster_id for cluster_id, end in sorted(v, key=operator.itemgetter(1), reverse=True)]
        for k, v in end_map.items():
            end_map[k] = [cluster_id for cluster_id, start in sorted(v, key=operator.itemgetter(1), reverse=True)]
        prediction_map[doc_key] = (start_map, end_map, word_map)

    word_index = 0
    new_conll = ''
    for line in input_file.split('\n'):
        if line.startswith("#begin"):
            new_conll += line + '\n'
            continue
        elif line.startswith("#end document"):
            new_conll += line
            continue
        else:
            row = line.split()
            if len(row) == 0:
                new_conll += '\n'
                continue

            glen = 0
            for l in row[:-1]:
                glen += len(l)
            glen += len(row[:-1])

            doc_key = 'bc' + '{}_{}'.format(row[0], row[1])
            start_map, end_map, word_map = prediction_map[doc_key]
            coref_list = []
            if word_index in end_map:
                for cluster_id in end_map[word_index]:
                    coref_list.append("{})".format(cluster_id))
            if word_index in word_map:
                for cluster_id in word_map[word_index]:
                    coref_list.append("({})".format(cluster_id))
            if word_index in start_map:
                for cluster_id in start_map[word_index]:
                    coref_list.append("({}".format(cluster_id))

            if len(coref_list) == 0:
                row[-1] = "-"
            else:
                row[-1] = "|".join(coref_list)

            word_index += 1

            line = line[:glen] + row[-1]
            new_conll += line + '\n'

    return new_conll


# ---------------------------------------------- build data utils ---------------------------------------------------


class watcher():
    """
    A class that ensures that there are no violations in the structure of mentions.
    """

    def __init__(self):
        self.mentions = 0

    def mentions_closed(self, s):
        s = s.split('|')
        for x in s:
            if x[0] == '(' and x[-1] != ')':
                self.mentions += 1
            elif x[0] != '(' and x[-1] == ')':
                self.mentions -= 1

        if self.mentions != 0:
            return False
        else:
            return True


def RuCoref2CoNLL(path, out_path, language='russian'):
    """
    RuCor corpus files are converted to standard conll format.
    Args:
        path: path to the RuCorp dataset
        out_path: -
        language: language of dataset

    Returns: Nothing

    """
    data = {"doc_id": [],
            "part_id": [],
            "word_number": [],
            "word": [],
            "part_of_speech": [],
            "parse_bit": [],
            "lemma": [],
            "sense": [],
            "speaker": [],
            "entiti": [],
            "predict": [],
            "coref": []}

    part_id = '0'
    speaker = 'spk1'
    sense = '-'
    entiti = '-'
    predict = '-'

    tokens_ext = "txt"
    groups_ext = "txt"
    tokens_fname = "Tokens"
    groups_fname = "Groups"

    tokens_path = os.path.join(path, ".".join([tokens_fname, tokens_ext]))
    groups_path = os.path.join(path, ".".join([groups_fname, groups_ext]))
    print('Convert rucoref corpus into conll format ...')
    start = time.time()
    coref_dict = {}
    with open(groups_path, "r") as groups_file:
        for line in groups_file:
            doc_id, variant, group_id, chain_id, link, shift, lens, content, tk_shifts, attributes, head, hd_shifts = \
                line[:-1].split('\t')

            if doc_id not in coref_dict:
                coref_dict[doc_id] = {'unos': defaultdict(list), 'starts': defaultdict(list), 'ends': defaultdict(list)}

            if len(tk_shifts.split(',')) == 1:
                coref_dict[doc_id]['unos'][shift].append(chain_id)
            else:
                tk = tk_shifts.split(',')
                coref_dict[doc_id]['starts'][tk[0]].append(chain_id)
                coref_dict[doc_id]['ends'][tk[-1]].append(chain_id)
        groups_file.close()

    # Write conll structure
    with open(tokens_path, "r") as tokens_file:
        k = 0
        doc_name = '0'
        for line in tokens_file:
            doc_id, shift, length, token, lemma, gram = line[:-1].split('\t')

            if doc_id == 'doc_id':
                continue

            if doc_id != doc_name:
                doc_name = doc_id
                w = watcher()
                k = 0

            data['word'].append(token)
            data['doc_id'].append(doc_id)
            data['part_id'].append(part_id)
            data['lemma'].append(lemma)
            data['sense'].append(sense)
            data['speaker'].append(speaker)
            data['entiti'].append(entiti)
            data['predict'].append(predict)
            data['parse_bit'].append('-')

            opens = coref_dict[doc_id]['starts'][shift] if shift in coref_dict[doc_id]['starts'] else []
            ends = coref_dict[doc_id]['ends'][shift] if shift in coref_dict[doc_id]['ends'] else []
            unos = coref_dict[doc_id]['unos'][shift] if shift in coref_dict[doc_id]['unos'] else []
            s = []
            s += ['({})'.format(el) for el in unos]
            s += ['({}'.format(el) for el in opens]
            s += ['{})'.format(el) for el in ends]
            s = '|'.join(s)
            if len(s) == 0:
                s = '-'
                data['coref'].append(s)
            else:
                data['coref'].append(s)

            closed = w.mentions_closed(s)
            if gram == 'SENT' and not closed:
                data['part_of_speech'].append('.')
                data['word_number'].append(k)
                k += 1

            elif gram == 'SENT' and closed:
                data['part_of_speech'].append(gram)
                data['word_number'].append(k)
                k = 0
            else:
                data['part_of_speech'].append(gram)
                data['word_number'].append(k)
                k += 1

        tokens_file.close()

    # Write conll structure in file
    conll = os.path.join(out_path, ".".join([language, 'v4_conll']))
    with open(conll, 'w') as CoNLL:
        for i in tqdm(range(len(data['doc_id']))):
            if i == 0:
                CoNLL.write('#begin document ({}); part {}\n'.format(data['doc_id'][i], data["part_id"][i]))
                CoNLL.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(data['doc_id'][i],
                                                                                       data["part_id"][i],
                                                                                       data["word_number"][i],
                                                                                       data["word"][i],
                                                                                       data["part_of_speech"][i],
                                                                                       data["parse_bit"][i],
                                                                                       data["lemma"][i],
                                                                                       data["sense"][i],
                                                                                       data["speaker"][i],
                                                                                       data["entiti"][i],
                                                                                       data["predict"][i],
                                                                                       data["coref"][i]))
            elif i == len(data['doc_id']) - 1:
                CoNLL.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(data['doc_id'][i],
                                                                                       data["part_id"][i],
                                                                                       data["word_number"][i],
                                                                                       data["word"][i],
                                                                                       data["part_of_speech"][i],
                                                                                       data["parse_bit"][i],
                                                                                       data["lemma"][i],
                                                                                       data["sense"][i],
                                                                                       data["speaker"][i],
                                                                                       data["entiti"][i],
                                                                                       data["predict"][i],
                                                                                       data["coref"][i]))
                CoNLL.write('\n')
                CoNLL.write('#end document\n')
            else:
                if data['doc_id'][i] == data['doc_id'][i + 1]:
                    CoNLL.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(data['doc_id'][i],
                                                                                           data["part_id"][i],
                                                                                           data["word_number"][i],
                                                                                           data["word"][i],
                                                                                           data["part_of_speech"][i],
                                                                                           data["parse_bit"][i],
                                                                                           data["lemma"][i],
                                                                                           data["sense"][i],
                                                                                           data["speaker"][i],
                                                                                           data["entiti"][i],
                                                                                           data["predict"][i],
                                                                                           data["coref"][i]))
                    if data["word_number"][i + 1] == 0:
                        CoNLL.write('\n')
                else:
                    CoNLL.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(data['doc_id'][i],
                                                                                           data["part_id"][i],
                                                                                           data["word_number"][i],
                                                                                           data["word"][i],
                                                                                           data["part_of_speech"][i],
                                                                                           data["parse_bit"][i],
                                                                                           data["lemma"][i],
                                                                                           data["sense"][i],
                                                                                           data["speaker"][i],
                                                                                           data["entiti"][i],
                                                                                           data["predict"][i],
                                                                                           data["coref"][i]))
                    CoNLL.write('\n')
                    CoNLL.write('#end document\n')
                    CoNLL.write('#begin document ({}); part {}\n'.format(data['doc_id'][i + 1], data["part_id"][i + 1]))

    print('End of convertion. Time - {}'.format(time.time() - start))
    return None


def split_doc(inpath, outpath, language='russian'):
    """
    It splits one large conll file containing the entire RuCorp dataset into many separate documents.
    Args:
        inpath: -
        outpath: -
        language: -

    Returns: Nothing

    """
    # split massive conll file to many little

    print('Start of splitting ...')
    with open(inpath, 'r+') as f:
        lines = f.readlines()
        f.close()
    set_ends = []
    k = 0
    print('Splitting conll document ...')
    for i in range(len(lines)):
        if lines[i].startswith('#begin'):
            doc_num = lines[i].split(' ')[2][1:-2]
        elif lines[i] == '#end document\n':
            set_ends.append([k, i, doc_num])
            k = i + 1
    for i in range(len(set_ends)):
        cpath = os.path.join(outpath, ".".join([str(set_ends[i][2]), language, 'v4_conll']))
        with open(cpath, 'w') as c:
            for j in range(set_ends[i][0], set_ends[i][1] + 1):
                if lines[j] == '#end document\n':
                    c.write(lines[j][:-1])
                else:
                    c.write(lines[j])
            c.close()

    del lines
    print('Splitts {} docs in {}.'.format(len(set_ends), outpath))
    del set_ends
    del k

    return None


def train_test_split(inpath, train, test, split, random_seed):
    """
    RuCor doesn't provide train/test data splitting, it makes random splitting.
    Args:
        inpath: path to data
        train: path to train folder
        test: path to test folder
        split: int, split ratio
        random_seed: seed for random module

    Returns:

    """
    print('Start train-test splitting ...')
    z = os.listdir(inpath)
    doc_split = ShuffleSplit(1, test_size=split, random_state=random_seed)
    for train_indeses, test_indeses in doc_split.split(z):
        train_set = [z[i] for i in sorted(list(train_indeses))]
        test_set = [z[i] for i in sorted(list(test_indeses))]
    for x in train_set:
        shutil.move(os.path.join(inpath, x), os.path.join(train, x))
    for x in test_set:
        shutil.move(os.path.join(inpath, x), os.path.join(test, x))
    print('End train-test splitts.')
    return None


def get_all_texts_from_tokens_file(tokens_path, out_path):
    """
    Creates file with pure text from RuCorp dataset.
    Args:
        tokens_path: -
        out_path: -

    Returns: Nothing

    """
    lengths = {}
    # determine number of texts and their lengths
    with open(tokens_path, "r") as tokens_file:
        for line in tokens_file:
            doc_id, shift, length, token, lemma, gram = line[:-1].split('\t')
            try:
                doc_id, shift, length = map(int, (doc_id, shift, length))
                lengths[doc_id] = shift + length
            except ValueError:
                pass

    texts = {doc_id: [' '] * length for (doc_id, length) in lengths.items()}
    # read texts
    with open(tokens_path, "r") as tokens_file:
        for line in tokens_file:
            doc_id, shift, length, token, lemma, gram = line[:-1].split('\t')
            try:
                doc_id, shift, length = map(int, (doc_id, shift, length))
                texts[doc_id][shift:shift + length] = token
            except ValueError:
                pass
    for doc_id in texts:
        texts[doc_id] = "".join(texts[doc_id])

    with open(out_path, "w") as out_file:
        for doc_id in texts:
            out_file.write(texts[doc_id])
            out_file.write("\n")
    return None


def get_char_vocab(input_filename, output_filename):
    """
    Gets chars dictionary from text, and write it into output_filename.
    Args:
        input_filename: -
        output_filename: -

    Returns: Nothing

    """
    data = open(input_filename, "r").read()
    vocab = sorted(list(set(data)))

    with open(output_filename, 'w') as f:
        for c in vocab:
            f.write(u"{}\n".format(c))
    print("[Wrote {} characters to {}] ...".format(len(vocab), output_filename))


def conll2dict(conll, iter_id=None, agent=None, mode='train', doc=None, epoch_done=False):
    """
    Opens the document, reads it, and adds its contents as a string to the dictionary.
    Args:
        conll: path to the conll file
        iter_id: number of operations
        agent: agent name
        mode: train/valid mode
        doc: document name
        epoch_done: flag

    Returns: dict { 'iter_id': iter_id,
                    'id': agent,
                    'epoch_done': epoch_done,
                    'mode': mode,
                    'doc_name': doc
                    'conll_str': s}

    """
    data = {'iter_id': iter_id,
            'id': agent,
            'epoch_done': epoch_done,
            'mode': mode,
            'doc_name': doc}

    with open(conll, 'r', encoding='utf8') as f:
        s = f.read()
        data['conll_str'] = s
        f.close()
    return data


def make_summary(value_dict):
    """Make tf.Summary for tensorboard"""
    return tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k, v in value_dict.items()])


def summary(value_dict, global_step, writer):
    """Make tf.Summary for tensorboard"""
    summary = tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k, v in value_dict.items()])
    writer.add_summary(summary, global_step)
    return None


def extract_data(infile):
    """
    Extract useful information from conll file and write it in special dict structure.

    {'doc_name': name,
     'parts': {'0': {'text': text,
                    'chains': {'1421': {'chain_id': 'bfdb6d6f-6529-46af-bd39-cf65b96479b5',
                                        'mentions': [{'sentence_id': 0,
                                                    'start_id': 0,
                                                    'end_id': 4,
                                                    'start_line_id': 1,
                                                    'end_line_id': 5,
                                                    'mention': 'Отличный отель для бюджетного отдыха',
                                                    'POS': ['Afpmsnf', 'Ncmsnn', 'Sp-g', 'Afpmsgf', 'Ncmsgn'],
                                                    'mention_id': 'f617f5ca-ed93-49aa-8eae-21b672c5a9df',
                                                    'mention_index': 0}
                                        , ... }}}}}

    Args:
        infile: path to ground conll file

    Returns:
        dict with an alternative mention structure
    """
    # filename, parts: [{text: [sentence, ...], chains}, ...]
    data = {'doc_name': None, 'parts': dict()}
    with open(infile, 'r', encoding='utf8') as fin:
        current_sentence = []
        current_sentence_pos = []
        sentence_id = 0
        opened_labels = dict()
        # chain: chain_id -> list of mentions
        # mention: sent_id, start_id, end_id, mention_text, mention_pos
        chains = dict()
        part_id = None
        mention_index = 0
        for line_id, line in enumerate(fin.readlines()):
            # if end document part then add chains to data
            if '#end document' == line.strip():
                assert part_id is not None, print('line_id: ', line_id)
                data['parts'][part_id]['chains'] = chains
                chains = dict()
                sentence_id = 0
                mention_index = 0
            # start or end of part
            if '#' == line[0]:
                continue
            line = line.strip()
            # empty line between sentences
            if len(line) == 0:
                assert len(current_sentence) != 0
                assert part_id is not None
                if part_id not in data['parts']:
                    data['parts'][part_id] = {'text': [], 'chains': dict()}
                data['parts'][part_id]['text'].append(' '.join(current_sentence))
                current_sentence = []
                current_sentence_pos = []
                sentence_id += 1
            else:
                # line with data
                line = line.split()
                assert len(line) >= 3, print('assertion:', line_id, len(line), line)
                current_sentence.append(line[3])
                doc_name, part_id, word_id, word, ref = line[0], int(line[1]), int(line[2]), line[3], line[-1]
                pos, = line[4],
                current_sentence_pos.append(pos)

                if data['doc_name'] is None:
                    data['doc_name'] = doc_name

                if ref != '-':
                    # we have labels like (322|(32)
                    ref = ref.split('|')
                    # get all opened references and set them to current word
                    for i in range(len(ref)):
                        ref_id = int(ref[i].strip('()'))
                        if ref[i][0] == '(':
                            if ref_id in opened_labels:
                                opened_labels[ref_id].append((word_id, line_id))
                            else:
                                opened_labels[ref_id] = [(word_id, line_id)]

                        if ref[i][-1] == ')':
                            assert ref_id in opened_labels
                            start_id, start_line_id = opened_labels[ref_id].pop()

                            if ref_id not in chains:
                                chains[ref_id] = {
                                    'chain_id': str(uuid.uuid4()),
                                    'mentions': [],
                                }

                            chains[ref_id]['mentions'].append({
                                'sentence_id': sentence_id,
                                'start_id': start_id,
                                'end_id': word_id,
                                'start_line_id': start_line_id,
                                'end_line_id': line_id,
                                'mention': ' '.join(current_sentence[start_id:word_id + 1]),
                                'POS': current_sentence_pos[start_id:word_id + 1],
                                'mention_id': str(uuid.uuid4()),
                                'mention_index': mention_index,
                            })
                            assert len(current_sentence[start_id:word_id + 1]) > 0, (doc_name, sentence_id, start_id,
                                                                                     word_id, current_sentence)
                            mention_index += 1
    return data


def ana_doc_score(g_file, p_file):
    """
    The function takes two files at the input, analyzes them and calculates anaphora score. A weak criterion
    for antecedent identification is used.
    In the document, all the pronouns are used.

    Args:
        g_file: path to ground truth conll file
        p_file: path to predicted conll file

    Returns:
        precision, recall, F1-metrics
    """

    main = {}

    g_data = extract_data(g_file)
    p_data = extract_data(p_file)
    pnames = list(p_data['parts'][0]['chains'].keys())
    gnames = list(g_data['parts'][0]['chains'].keys())

    if len(pnames) != len(gnames):
        print('In documents, a different number of clusters.'
              ' {0} in gold file and {1} in predicted file.'.format(len(gnames), len(pnames)))
    # else:
    #     print('The documents have the same number of clusters.')

    for x in [p_data, g_data]:
        for y in x['parts'][0]['chains'].keys():
            for z in x['parts'][0]['chains'][y]['mentions']:
                if not z['start_line_id'] != z['end_line_id'] and z['POS'][0].startswith('P'):
                    c = (z['start_line_id'], z['end_line_id'])
                    if (z['start_line_id'], z['end_line_id']) not in main:
                        main[c] = dict()
                        main[c]['pred'] = list()
                        main[c]['gold'] = list()
                        if x is g_data:
                            main[c]['g_clus'] = gnames.index(y)
                            main[c]['g_clus_real'] = y
                        else:
                            main[c]['p_clus'] = pnames.index(y)
                            main[c]['p_clus_real'] = y
                    else:
                        if x is g_data:
                            main[c]['g_clus'] = gnames.index(y)
                            main[c]['g_clus_real'] = y
                        else:
                            main[c]['p_clus'] = pnames.index(y)
                            main[c]['p_clus_real'] = y

    for x in main:
        try:
            for y in g_data['parts'][0]['chains'][main[x]['g_clus_real']]['mentions']:
                if y['start_line_id'] < x[0] and y['end_line_id'] < x[1]:
                    main[x]['gold'].append((y['start_line_id'], y['end_line_id']))
                elif y['start_line_id'] < x[0] and y['end_line_id'] == x[1]:
                    main[x]['gold'].append((y['start_line_id'], y['end_line_id']))
        except KeyError:
            print("In gold file {0} absent cluster {1}".format(g_file, x))
            continue

        try:
            for y in p_data['parts'][0]['chains'][main[x]['p_clus_real']]['mentions']:
                if y['start_line_id'] < x[0] and y['end_line_id'] < x[1]:
                    main[x]['pred'].append((y['start_line_id'], y['end_line_id']))
                elif y['start_line_id'] < x[0] and y['end_line_id'] == x[1]:
                    main[x]['pred'].append((y['start_line_id'], y['end_line_id']))
        except KeyError:
            print("In predicted file {0} absent cluster {1}".format(p_file, x))
            continue

    # compute F1 score
    score = 0
    fn = 0
    fp = 0

    for x in main:
        k = 0
        if len(main[x]['pred']) != 0 and len(main[x]['gold']) != 0:
            for y in main[x]['pred']:
                if y in main[x]['gold']:
                    score += 1
                    k += 1
                    continue
            if k == 0:
                if main[x]['g_clus'] == main[x]['p_clus']:
                    fn += 1
                    # print('fn', fn)
                    # print(main[x])
                else:
                    fp += 1
                    # print('fp', fp)
                    # print(main[x])
        else:
            continue

    precision = score / (score + fp)
    recall = score / (score + fn)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def true_ana_score(g_file, p_file):
    main = {}

    g_data = extract_data(g_file)
    p_data = extract_data(p_file)
    pnames = list(p_data['parts'][0]['chains'].keys())
    gnames = list(g_data['parts'][0]['chains'].keys())

    if len(pnames) != len(gnames):
        print('In documents, a different number of clusters.'
              ' {0} in gold file and {1} in predicted file.'.format(len(gnames), len(pnames)))
    # else:
    #     print('The documents have the same number of clusters.')

    for x in [p_data, g_data]:
        for y in x['parts'][0]['chains'].keys():
            for z in x['parts'][0]['chains'][y]['mentions']:
                if not z['start_line_id'] != z['end_line_id'] and z['POS'][0].startswith('P'):
                    c = (z['start_line_id'], z['end_line_id'])
                    if (z['start_line_id'], z['end_line_id']) not in main:
                        main[c] = dict()
                        main[c]['pred'] = list()
                        main[c]['gold'] = list()
                        if x is g_data:
                            main[c]['g_clus'] = gnames.index(y)
                            main[c]['g_clus_real'] = y
                        else:
                            main[c]['p_clus'] = pnames.index(y)
                            main[c]['p_clus_real'] = y
                    else:
                        if x is g_data:
                            main[c]['g_clus'] = gnames.index(y)
                            main[c]['g_clus_real'] = y
                        else:
                            main[c]['p_clus'] = pnames.index(y)
                            main[c]['p_clus_real'] = y

    for x in main:
        try:
            for y in g_data['parts'][0]['chains'][main[x]['g_clus_real']]['mentions']:
                if y['start_line_id'] < x[0] and y['end_line_id'] < x[1]:
                    main[x]['gold'].append((y['start_line_id'], y['end_line_id']))
                elif y['start_line_id'] < x[0] and y['end_line_id'] == x[1]:
                    main[x]['gold'].append((y['start_line_id'], y['end_line_id']))
        except KeyError:
            print("In gold file {0} absent cluster {1}".format(g_file, x))
            continue

        try:
            for y in p_data['parts'][0]['chains'][main[x]['p_clus_real']]['mentions']:
                if y['start_line_id'] < x[0] and y['end_line_id'] < x[1]:
                    main[x]['pred'].append((y['start_line_id'], y['end_line_id']))
                elif y['start_line_id'] < x[0] and y['end_line_id'] == x[1]:
                    main[x]['pred'].append((y['start_line_id'], y['end_line_id']))
        except KeyError:
            print("In predicted file {0} absent cluster {1}".format(p_file, x))
            continue

    # compute F1 score
    tp = 0
    fn = 0
    fp = 0
    prec = list()
    rec = list()
    f_1 = list()

    for x in main:
        if len(main[x]['pred']) != 0 and len(main[x]['gold']) != 0:
            for y in main[x]['pred']:
                if y in main[x]['gold']:
                    tp += 1
                else:
                    fp += 1
            for y in main[x]['gold']:
                if y not in main[x]['pred']:
                    fn += 1

        if (tp + fp) == 0:
            p = 0
        else:
            p = tp / (tp + fp)

        if (tp + fn) == 0:
            r = 0
        else:
            r = tp / (tp + fn)

        if (p + r) != 0:
            f = 2 * p * r / (p + r)
        else:
            f = 0

        prec.append(p)
        rec.append(r)
        f_1.append(f)

    precision = np.mean(prec)
    recall = np.mean(rec)
    f1 = np.mean(f_1)

    return precision, recall, f1


def pos_ana_score(g_file, p_file):
    """
    Anaphora scores from gold and predicted files.
    A weak criterion for antecedent identification is used. In documents only used 3d-person pronouns,
    reflexive pronouns and relative pronouns.

    Args:
        g_file: path to ground truth conll file
        p_file: path to predicted conll file

    Returns:
        precision, recall, F1-metrics
    """

    main = {}
    reflexive_p = ['себя', 'себе', 'собой', 'собою', 'свой', 'своя', 'своё', 'свое', 'свои',
                   'своего', 'своей', 'своего', 'своих', 'своему', 'своей', 'своему',
                   'своим', 'своего', 'свою', 'своего', 'своих', 'своим', 'своей', 'своим',
                   'своими', 'своём', 'своем', 'своей', 'своём', 'своем', 'своих']
    relative_p = ['кто', 'что', 'какой', 'каков', 'который', 'чей', 'сколько', 'где', 'куда', 'когда',
                  'откуда', 'почему', 'зачем', 'как']

    g_data = extract_data(g_file)
    p_data = extract_data(p_file)
    pnames = list(p_data['parts'][0]['chains'].keys())
    gnames = list(g_data['parts'][0]['chains'].keys())

    if len(pnames) != len(gnames):
        print('In documents, a different number of clusters.'
              ' {0} in gold file and {1} in predicted file.'.format(len(gnames), len(pnames)))
    # else:
    #     print('The documents have the same number of clusters.')

    for x in [p_data, g_data]:
        for y in x['parts'][0]['chains'].keys():
            for z in x['parts'][0]['chains'][y]['mentions']:
                if not z['start_line_id'] != z['end_line_id']:
                    if z['POS'][0].startswith('P-3'):
                        c = (z['start_line_id'], z['end_line_id'])
                        if (z['start_line_id'], z['end_line_id']) not in main:
                            main[c] = dict()
                            main[c]['pred'] = list()
                            main[c]['gold'] = list()
                            if x is g_data:
                                main[c]['g_clus'] = gnames.index(y)
                                main[c]['g_clus_real'] = y
                            else:
                                main[c]['p_clus'] = pnames.index(y)
                                main[c]['p_clus_real'] = y
                        else:
                            if x is g_data:
                                main[c]['g_clus'] = gnames.index(y)
                                main[c]['g_clus_real'] = y
                            else:
                                main[c]['p_clus'] = pnames.index(y)
                                main[c]['p_clus_real'] = y

                    elif z['mention'] in relative_p:
                        c = (z['start_line_id'], z['end_line_id'])
                        if (z['start_line_id'], z['end_line_id']) not in main:
                            main[c] = dict()
                            main[c]['pred'] = list()
                            main[c]['gold'] = list()
                            if x is g_data:
                                main[c]['g_clus'] = gnames.index(y)
                                main[c]['g_clus_real'] = y
                            else:
                                main[c]['p_clus'] = pnames.index(y)
                                main[c]['p_clus_real'] = y
                        else:
                            if x is g_data:
                                main[c]['g_clus'] = gnames.index(y)
                                main[c]['g_clus_real'] = y
                            else:
                                main[c]['p_clus'] = pnames.index(y)
                                main[c]['p_clus_real'] = y

                    elif z['mention'] in reflexive_p:
                        c = (z['start_line_id'], z['end_line_id'])
                        if (z['start_line_id'], z['end_line_id']) not in main:
                            main[c] = dict()
                            main[c]['pred'] = list()
                            main[c]['gold'] = list()
                            if x is g_data:
                                main[c]['g_clus'] = gnames.index(y)
                                main[c]['g_clus_real'] = y
                            else:
                                main[c]['p_clus'] = pnames.index(y)
                                main[c]['p_clus_real'] = y
                        else:
                            if x is g_data:
                                main[c]['g_clus'] = gnames.index(y)
                                main[c]['g_clus_real'] = y
                            else:
                                main[c]['p_clus'] = pnames.index(y)
                                main[c]['p_clus_real'] = y

    for x in main:
        try:
            for y in g_data['parts'][0]['chains'][main[x]['g_clus_real']]['mentions']:
                if y['start_line_id'] < x[0] and y['end_line_id'] < x[1]:
                    main[x]['gold'].append((y['start_line_id'], y['end_line_id']))
                elif y['start_line_id'] < x[0] and y['end_line_id'] == x[1]:
                    main[x]['gold'].append((y['start_line_id'], y['end_line_id']))
        except KeyError:
            print("In gold file {0} absent cluster {1}".format(g_file, x))
            continue

        try:
            for y in p_data['parts'][0]['chains'][main[x]['p_clus_real']]['mentions']:
                if y['start_line_id'] < x[0] and y['end_line_id'] < x[1]:
                    main[x]['pred'].append((y['start_line_id'], y['end_line_id']))
                elif y['start_line_id'] < x[0] and y['end_line_id'] == x[1]:
                    main[x]['pred'].append((y['start_line_id'], y['end_line_id']))
        except KeyError:
            print("In predicted file {0} absent cluster {1}".format(p_file, x))
            continue

    # compute F1 score
    tp = 0
    fn = 0
    fp = 0
    prec = list()
    rec = list()
    f_1 = list()

    for x in main:
        if len(main[x]['pred']) != 0 and len(main[x]['gold']) != 0:
            for y in main[x]['pred']:
                if y in main[x]['gold']:
                    tp += 1
                else:
                    fp += 1
            for y in main[x]['gold']:
                if y not in main[x]['pred']:
                    fn += 1

        if (tp + fp) == 0:
            p = 0
        else:
            p = tp / (tp + fp)

        if (tp + fn) == 0:
            r = 0
        else:
            r = tp / (tp + fn)

        if (p + r) == 0:
            f = 0
        else:
            f = 2 * p * r / (p + r)

        prec.append(p)
        rec.append(r)
        f_1.append(f)

    precision = np.mean(prec)
    recall = np.mean(rec)
    f1 = np.mean(f_1)

    return precision, recall, f1


def anaphora_score(keys_path, predicts_path, stype):
    """
    Anaphora scores predicted files.
    A weak criterion for antecedent identification is used.
    The function takes the input to the path to the folder containing the key files, and the path to the folder
    containing the predicted files and the scorer type. It calculates the anaphora score for each file in folders,
    and averages the results. The number of files in the packs must be the same, otherwise the function
    will generate an error.

    Args:
        keys_path: path to ground truth conll files
        predicts_path: path to predicted conll files
        stype: a trigger that selects the type of scorer to be used

    Returns:
        dict with scores
    """
    pred_files = list(filter(lambda x: x.endswith('conll'), os.listdir(predicts_path)))

    # assert len(key_files) == len(pred_files), ('The number of visible files is not equal.')

    results = dict()
    results['precision'] = list()
    results['recall'] = list()
    results['F1'] = list()

    result = dict()

    for file in tqdm(pred_files):
        predict_file = os.path.join(predicts_path, file)
        gold_file = os.path.join(keys_path, file)

        if stype == 'full':
            p, r, f1 = true_ana_score(gold_file, predict_file)
        elif stype == 'semi':
            p, r, f1 = ana_doc_score(gold_file, predict_file)
        elif stype == 'pos':
            p, r, f1 = pos_ana_score(gold_file, predict_file)
        else:
            raise ValueError('Non supported type of anaphora scorer. {}'.format(stype))

        results['precision'].append(p)
        results['recall'].append(r)
        results['F1'].append(f1)

    result['precision'] = np.mean(results['precision'])
    result['recall'] = np.mean(results['recall'])
    result['F1'] = np.mean(results['F1'])

    return result


def build_coref_data(opt, version='1.3'):
    """prepares datasets and other dependencies for CoreferenceTeacher"""
    # get path to data directory and create folders tree
    dpath = join(opt['datapath'])
    # define version if any, and languages
    dpath = join(dpath, 'coreference' + version)
    print(f'[building data: {dpath} ... ]')

    # check if data had been previously built
    os.makedirs(dpath)

    # Build the folders tree
    os.makedirs(join(dpath, 'scorer'))
    os.makedirs(join(dpath, 'train'))
    os.makedirs(join(dpath, 'valid'))

    # urls
    dataset_url = 'http://rucoref.maimbava.net/files/rucoref_29.10.2015.zip'
    scorer_url = 'http://conll.cemantix.org/download/reference-coreference-scorers.v8.01.tar.gz'

    # download the conll-2012 scorer v 8.1
    start = time.time()
    print('[Download the conll-2012 scorer]...')
    download_decompress(scorer_url, join(dpath, 'scorer'))
    print('[Scorer was dawnloads]...')

    # download dataset
    print('[Download the rucoref dataset]...')
    os.makedirs(join(dpath, 'rucoref_29.10.2015'))
    download_decompress(dataset_url, join(dpath, 'rucoref_29.10.2015'))
    print('End of download: time - {}'.format(time.time() - start))

    # Convertation rucorpus files in conll files
    conllpath = join(dpath, 'ru_conll')
    os.makedirs(conllpath)
    RuCoref2CoNLL(join(dpath, 'rucoref_29.10.2015'), conllpath)

    # splits conll files
    start = time.time()
    conlls = join(dpath, 'ru_conlls')
    os.makedirs(conlls)
    split_doc(join(conllpath, 'russian.v4_conll'), conlls)
    os.remove(conllpath)

    # create train valid test partitions
    # train_test_split(conlls,dpath,opt['split'],opt['random-seed'])
    train_test_split(conlls, join(dpath, 'train'), join(dpath, 'valid'), 0.2, None)
    copytree(join(dpath, 'valid'), join(dpath, 'test'))
    os.remove(conlls)
    os.remove(join(dpath, 'rucoref_29.10.2015'))
    print('End of data splitting. Time - {}'.format(time.time() - start))
    print('[Datasets done.]')
    return None
