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

import collections

import fastText
import numpy as np

from deeppavlov.core.commands.utils import expand_path


# todo delete after writing vocabs
def load_char_dict(char_vocab_path):
    """Load char dict from file"""
    vocab = [u"<unk>"]
    with expand_path(char_vocab_path).open('r') as f:
        vocab.extend(c.strip() for c in f.readlines())
    char_dict = collections.defaultdict(int)
    char_dict.update({c: i for i, c in enumerate(sorted(set(vocab)))})
    return char_dict


# todo delete after writing embedders
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
        embedding_dict = fastText.load_model(str(expand_path(embedding_path)))
    else:
        raise ValueError('Not supported embeddings format {}'.format(embedding_format))
    print("Done loading word embeddings.")
    return embedding_dict
