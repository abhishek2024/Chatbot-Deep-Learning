# Copyright 1017 Neural Networks and Deep Learning lab, MIPT
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

import pickle
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Tuple, List, Union

import numpy as np
from nltk import word_tokenize
from tqdm import tqdm

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import download
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.estimator import Estimator

logger = get_logger(__name__)

@register('ner_rel_extractor')
class NerRelExtractor(Component):
    """ NerRelExtractor class is responsible for processing tokenized inputs.

        It extracts multiple relations for each word/bigram in a tokenized form.
    """

    def __init__(self, num_relations_limit: int = 10, num_word_per_relation_limit: int = 3, \
                 relation_dict_pickle: str = 'relations/word2vec_dict_ngrams.pickle', *args, **kwargs):
        self.num_relations_limit = num_relations_limit
        self.num_word_per_relation_limit = num_word_per_relation_limit
        self.relation_dict_pickle = expand_path(relation_dict_pickle)

        with open(self.relation_dict_pickle, 'rb') as handle:
            self.word_relations_dict = pickle.load(handle)

    def __call__(self, tokenized_lists: List[List[str]], *args, **kwargs):
        """ Returns for each tokenized input the list of tokenized relations
            Outputs List[List[List[List[str]]]]
        """
        max_sentence_len = max(len(sentence) for sentence in tokenized_lists)

        tokenized_relations = []

        zero_str = [["<UNK>" for i in range(self.num_word_per_relation_limit)] for j in range(self.num_relations_limit)]

        for sentence in tokenized_lists:
            list_of_words = []
            for word in sentence:
                if word.lower() in self.word_relations_dict:
                    tokenized = [self._split_phrase_into_concepts(relation) for relation in self._get_padded_relations(word)]
                    list_of_words.append(tokenized)
                else:
                    list_of_words.append(zero_str)

            tokenized_relations.append(list_of_words)
        
        #print(tokenized)
        #print(zero_str)


        #tokenized_relations = [[[self._split_phrase_into_concepts(relation)
        #                         for relation in self._get_padded_relations(word)]
        #                        for word in word_list if word.lower() in self.word_relations_dict]
        #                       for word_list in tokenized_lists]

        return tokenized_relations, max_sentence_len

    def _get_padded_relations(self, word):
        """ Given word, returns padded relations
        """

        relations = self.word_relations_dict[word.lower()][0:self.num_relations_limit]
        return relations

    def _split_phrase_into_concepts(self, phrase):
        """ Given a phrase, splits it into list of tokens
        """

        words = phrase.split()[:self.num_word_per_relation_limit]
        return words

@register('collapse')
class Collapse(Component):
    """
    Given nested list of 4 layes,
    returns list of 2 layers and corresponding
    list of lengths
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, tensor, *args, **kwargs):
        """

        Args:
            tensor: nested list of tokens
            *args:
            **kwargs:

        Returns:
            tensor_flat: flattened list consisting of 2 layers
            lens: a list with indices, needed for uncollapsing the list back to 4D
        """
        #print("tensor_length", len(tensor))
        
        tensor_flat = []
        indices = []

        for i in range(len(tensor)):
            cnt = 0
            dic = []
            tensor_flat_0 = []
            len_1 = len(tensor[i])

            for j in range(len_1):
                len_2 = len(tensor[i][j])

                for k in range(len_2):
                    len_3 = len(tensor[i][j][k])
                
                    for l in range(len_3):
                        cnt += 1
                        dic.append([j, k, l])
                        tensor_flat_0.append(tensor[i][j][k][l])
                
            indices.append(dic)
            tensor_flat.append(tensor_flat_0)
            
            #if i == 0:
            #    print("cnt", cnt)

        #print("element", tensor[0][0][0][0])

        #print("collapse_lens", len(tensor_flat[0]), len(indices[0]), tensor_flat[0][0])

        return tensor_flat, indices

@register('unfold')
class Unfold(Component):
    """
    Unfolds 2d list to 4d
    using indices
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, tensor_flat, indices, max_sentence_len, *args, **kwargs):
        tensor = []
        bs = len(tensor_flat)

        max_num_rel_tokens = 3
        max_num_rels = 10
        
        zero_pad = np.zeros((bs, max_sentence_len, max_num_rels, max_num_rel_tokens), dtype=int)
        
        #print("len_1_dim", len(tensor_flat), len(indices))        

        for i in range(len(tensor_flat)):
            #print("len_2_dim", len(tensor_flat[i]), len(indices[i]))        

            flattened_index = len(tensor_flat[i])
            for j in range(flattened_index):
                first, second, third = indices[i][j]
                zero_pad[i][first][second][third] = tensor_flat[i][j]

        return zero_pad
