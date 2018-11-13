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


from typing import Union

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.simple_vocab import SimpleVocabulary
from deeppavlov.models.embedders.fasttext_embedder import FasttextEmbedder
from deeppavlov.models.embedders.glove_embedder import GloVeEmbedder


@register('zero_shot_description_assembler')
class ZeroShotDescriptionEmbeddingAssembler:
    """Given a list of classes each equipped with description this class this class will assemble
    the matrix of mean embeddings of descriptions for each class

    Args:
        embedder: an instance of the class that convertes tokens to vectors.
            For example :class:`~deeppavlov.models.embedders.fasttext_embedder.FasttextEmbedder` or
            :class:`~deeppavlov.models.embedders.glove_embedder.GloVeEmbedder`
        vocab: instance of :class:`~deeppavlov.core.data.SimpleVocab`. tag vocabulary

    Attributes:
        dim: dimensionality of the embeddings (can be less than dimensionality of
            embeddings produced by `Embedder`.
    """

    def __init__(self,
                 embedder: Union[FasttextEmbedder, GloVeEmbedder],
                 tag_vocab: SimpleVocabulary,
                 dataset_name: str,
                 *args,
                 **kwargs) -> None:
        self.embedder = embedder
        if dataset_name == 'dstc2':
            self._descr = dict(area='area',
                               food='food',
                               pricerange='pricerange',
                               name='restaurant name')
        elif dataset_name == 'simm':
            self._descr = dict(movie='movie',
                               theatre_name='theatre name',
                               num_tickets='number of tickets',
                               date='date',
                               time='time')
        elif dataset_name == 'simr':
            self._descr = dict(time='time',
                               date='date',
                               restaurant_name='restaurant name',
                               num_people='number of people',
                               location='location',
                               price_range='price range',
                               category='category',
                               meal='meal',
                               rating='rating')
        else:
            raise RuntimeError('The dataset_name must be in: '
                               '{"dstc2", "simm", "simr"}, '
                               f'however, dataset_name = {dataset_name}')
        self.emb_mat = np.zeros([len(tag_vocab), self.dim])
        for n, tag in enumerate(tag_vocab):
            if tag != 'O':
                if tag.startswith('B-') or tag.startswith('I-'):
                    tag = tag[2:]
                embeddings = self.embedder([self._descr[tag].split()])[0]
                print(tag)
                print(embeddings)
                self.emb_mat[n] = np.mean(embeddings, axis=0)
        # import pdb; pdb.set_trace()
        print('Success')

    @property
    def dim(self):
        return self.embedder.dim

