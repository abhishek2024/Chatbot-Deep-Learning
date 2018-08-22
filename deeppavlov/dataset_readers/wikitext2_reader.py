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


from pathlib import Path

import pandas as pd
from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download, mark_done
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


@register('wikitext2_reader')
class Wikitext2DatasetReader(DatasetReader):
    """
    Class provides reading Wikitext-2 dataset
    """

    @overrides
    def read(self, data_path: str,
             *args, **kwargs) -> dict:
        """
        Read dataset from data_path directory

        Args:
            data_path: directory with files

        Returns:
            dictionary with types from data_types.
            Each field of dictionary is a list of tuples (x_i, y_i)
        """
        data_types = ["train", "valid", "test"]

        train_file = kwargs.get('train', 'wiki.train.tokens')

        if not Path(data_path, train_file).exists():
            raise Exception("data path {} does not exist or is empty, and download url parameter not specified!".format(data_path))

        data = {"train": [],
                "valid": [],
                "test": []}
        for data_type in data_types:
            file_name = kwargs.get(data_type, 'wiki.{}.tokens'.format(data_type))
            file = Path(data_path).joinpath(file_name)
            if file.exists():
                data[data_type] = self.read_wikitext2_file(file)
            else:
                log.warning("Cannot find {} file".format(file))

        return data

    def read_wikitext2_file(self, file_name):
        with open(file_name, "r") as f:
            df = f.read().splitlines()

        title_read = False
        body_read = False
        data_list = []

        body = ""
        title = None

        for row in df:
            if row != " ":
                if title_read:
                    body += "\n" + row
                    body_read = True
                else:
                    title_read = True
                    title = row
            else:
                if title_read is False and body_read is False:
                    continue
                elif title_read is True and body_read is False:
                    continue
                elif title_read is True and body_read is True:
                    data_list.append((body, title))
                    body_read = False
                    title_read = False
                    body = ""
                    title = None
                else:
                    continue

        return data_list
