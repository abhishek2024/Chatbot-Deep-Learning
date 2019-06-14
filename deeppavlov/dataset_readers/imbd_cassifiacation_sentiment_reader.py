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

from logging import getLogger
from pathlib import Path

import pandas as pd
from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download

log = getLogger(__name__)


@register('imdb_sentiment_reader')
class IMDBSentimentDatasetReader(DatasetReader):
    
    @overrides
    def read(self, data_path: str, url: str = None, *args, **kwargs) -> dict:
        data_types = ["train","test"]
        
#         train_file = kwargs.get('train', 'train.csv')
#         if not Path(data_path, train_file).exists():
#             if url is None:
#                 raise Exception("data path {} does not exist or is empty, and download url parameter not specified!".format(data_path))
#             log.info("Loading train data from {} to {}".format(url, data_path))
#             download(source_url=url, dest_file_path=Path(data_path, train_file)) #TODO: А что собственно будет читаться с сервака в этой точке?

        data = {"train": [],
                "valid": [],
                "test": []}
        data_labels = ["neg","pos"]
        
        for data_type in data_types:
            for data_label in data_labels :
                review_paths = [Path(data_path,data_type,data_label,review_path) 
                                for review_path in os.listdir(Path(data_path,data_type,data_label))]
                
                for review in review_paths:
                    with open(review, encoding='utf-8') as review_file:
                        review = review_file.read()
                        data[data_type].append((review,data_label))
                        
        quartil_boundary = int(len(data["test"]) / 4) #в данных сначала половина негативных, потом позитивные. положим в valid 1ю и 4ю квартиль
        data["valid"] = data["test"][:quartil_boundary] + data["test"][3*quartil_boundary:]
        data["test"] = data["test"][quartil_boundary:3*quartil_boundary]
        
        print("all was red")

        return data