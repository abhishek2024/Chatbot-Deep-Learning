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
from typing import Generator

from overrides import overrides
from sklearn.model_selection import train_test_split

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.core.common.log import get_logger


logger = get_logger(__name__)


@register('basic_neural_ranker_iterator')
class BasicNeuralRankerIterator(DataLearningIterator):

    def __init__(self, data,
                 seed: int = None, shuffle: bool = True,
                 *args, **kwargs):

        super().__init__(data, seed=seed, shuffle=shuffle)
        # all_data = data['train'] + data['valid']
        val, _ = train_test_split(data['valid'], test_size=0.5, shuffle=True)
        # val, test = train_test_split(val, test_size=0.5, shuffle=True)
        # self.train = tr
        # self.valid = val
        # self.test = test
        self.data = {'train': data['train'],
                     'valid': val}

    @overrides
    def get_instances(self, data_type: str = 'train') -> tuple:
        """
        Reformat data to x, y pairs, where x, y is a single dataset instance.
        Args:
            data_type (str): can be either 'train', 'test', or 'valid'
        Returns:
            x, y pairs
        """
        zero_len = len(self.data[data_type][0]['contexts']) - 1
        labels = [[1] + [0]*zero_len for _ in range(len(self.data[data_type]))]
        return tuple(zip(self.data, labels))

    @overrides
    def gen_batches(self, batch_size: int, data_type: str = 'train',
                    shuffle: bool = None) -> Generator:
        """Return a generator, which serves for generation of raw (no preprocessing such as tokenization)
        batches
        Args:
            batch_size (int): number of samples in batch
            data_type (str): can be either 'train', 'test', or 'valid'
            shuffle (bool): whether to shuffle dataset before batching
        Returns:
            batch_gen (Generator): a generator, that iterates through the part (defined by data_type) of the dataset
        """
        if shuffle is None:
            shuffle = self.shuffle

        data = self.data[data_type]
        data_len = len(data)

        if data_len == 0:
            return

        order = list(range(data_len))
        if shuffle:
            self.random.shuffle(order)

        if batch_size < 0:
            batch_size = data_len

        zero_len = len(data[0]['contexts']) - 1

        labels = [[1] + [0]*zero_len for _ in range(data_len)]

        batch_n = (data_len - 1) // batch_size + 1

        for i in range(batch_n):
            # logger.info(f"Processing batch {i} of {batch_n}")
            x_instance = [[data[o] for o in order[i * batch_size:(i + 1) * batch_size]]]
            res = tuple(zip(x_instance, labels))[0]
            yield res


