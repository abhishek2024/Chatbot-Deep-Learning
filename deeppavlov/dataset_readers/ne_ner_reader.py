from deeppavlov.core.data.utils import download_decompress

import deeppavlov.dataset_readers.ne_parser as ne_parser
from deeppavlov.core.data.dataset_reader import DatasetReader
from pathlib import Path
from deeppavlov.core.common.registry import register
import pickle


@register('ne_ner_reader')
class NENERDatasetReader(DatasetReader):
    def read(self, data_path: str):
        data_path = Path(data_path)
        ne_parser.slurp_NE5_annotated_data(data_path / 'train', 'train')
        ne_parser.slurp_NE5_annotated_data(data_path / 'test', 'test')
        ne_parser.slurp_NE5_annotated_data(data_path / 'valid', 'valid')
        train = ne_parser.get_supervised_data('train')
        train_filtered = []
        for sample in train:
            if max(len(tok) for tok in sample[0]) < 100:
                train_filtered.append(sample)

        test = ne_parser.get_supervised_data('test')
        valid = [s for s in ne_parser.get_supervised_data('valid') if max(len(tok) for tok in s[0]) < 100]
        return dict(train=train_filtered, valid=valid, test=test)
