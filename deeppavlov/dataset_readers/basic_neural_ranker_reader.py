from pathlib import Path

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.common.file import read_json


@register('basic_neural_ranker_reader')
class BasicNeuralRankerReader(DatasetReader):

    def read(self, data_path):
        train_fname = Path(data_path) / 'train-v1.1_negative_samples.json'
        valid_fname = Path(data_path) / 'dev-v1.1_negative_samples.json'

        data = {"train": read_json(train_fname),
                "valid": read_json(valid_fname)}

        return data
