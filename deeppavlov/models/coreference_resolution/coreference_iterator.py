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

from typing import Dict, Tuple, Iterator
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator


class CorefIterator(DataLearningIterator):
    """
    Dataset iterator for learning models, e. g. neural networks.
    """

    def gen_batches(self, data_type: str = 'train', batch_size: int = 1,
                    shuffle: bool = None) -> Iterator[Tuple[Dict, Dict]]:
        """Generate batches of inputs and expected output to train neural networks

        Args:
            batch_size: number of samples in batch
            data_type: can be either 'train', 'test', or 'valid'
            shuffle: whether to shuffle dataset before batching

        Yields:
             a tuple of a batch of inputs and a batch of expected outputs
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

        for z in order:
            x = {"doc_key": z["doc_key"],
                 "sentences": z["sentences"],
                 "speakers": z["speakers"]}
            y = {"clusters": z["clusters"]}
            yield tuple(zip(x, y))
