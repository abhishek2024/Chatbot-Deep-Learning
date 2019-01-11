# Copyright 2018 Neural Networks and Deep Learning lab, MIPT
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
from typing import Dict, List, Union, Tuple, Optional

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import download_decompress, mark_done
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.data.dataset_reader import DatasetReader

import sys

WORD_COLUMN, POS_COLUMN, TAG_COLUMN = 1, 3, 5


log = get_logger(__name__)

def get_language(filepath: str) -> str:
    """Extracts language from typical UD filename
    """
    return filepath.split("-")[0]

def read_infile(infile: Union[Path, str], from_words=False,
                word_column: int = WORD_COLUMN, pos_column: int = POS_COLUMN,
                tag_column: int = TAG_COLUMN, max_sents: int = -1,
                read_only_words: bool = False) -> List[Tuple[List, Union[List, None]]]:
    """Reads input file in CONLL-U format

    Args:
        infile: a path to a file
        word_column: column containing words (default=1)
        pos_column: column containing part-of-speech labels (default=3)
        tag_column: column containing fine-grained tags (default=5)
        max_sents: maximal number of sents to read
        read_only_words: whether to read only words

    Returns:
        a list of sentences. Each item contains a word sequence and a tag sequence, which is ``None``
        in case ``read_only_words = True``
    """
    answer, curr_word_sent, curr_tag_sent = [], [], []
    if from_words:
        word_column, read_only_words = 0, True
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if line.startswith("#"):
                continue
            if line == "":
                if len(curr_word_sent) > 0:
                    if read_only_words:
                        curr_tag_sent = None
                    answer.append((curr_word_sent, curr_tag_sent))
                curr_tag_sent, curr_word_sent = [], []
                if len(answer) == max_sents:
                    break
                continue
            splitted = line.split("\t")
            index = splitted[0]
            if not from_words and not index.isdigit():
                continue
            curr_word_sent.append(splitted[word_column])
            if not read_only_words:
                pos, tag = splitted[pos_column], splitted[tag_column]
                tag = pos if tag == "_" else "{},{}".format(pos, tag)
                curr_tag_sent.append(tag)
        if len(curr_word_sent) > 0:
            if read_only_words:
                curr_tag_sent = None
            answer.append((curr_word_sent, curr_tag_sent))
    return answer


@register('morphotagger_dataset_reader')
class MorphotaggerDatasetReader(DatasetReader):
    """Class to read training datasets in UD format"""

    URL = 'http://files.deeppavlov.ai/datasets/UD2.0_source/'

    def read(self, data_path: Union[List, str],
             language: Optional[None] = None,
             data_types: Optional[List[str]] = None,
             **kwargs) -> Dict[str, List]:
        """Reads UD dataset from data_path.

        Args:
            data_path: can be either
                1. a directory containing files. The file for data_type 'mode'
                is then data_path / {language}-ud-{mode}.conllu
                2. a list of files, containing the same number of items as data_types
            language: a language to detect filename when it is not given
            data_types: which dataset parts among 'train', 'dev', 'test' are returned

        Returns:
            a dictionary containing dataset fragments (see ``read_infile``) for given data types
        """
        if data_types is None:
            data_types = ["train", "dev"]
        elif isinstance(data_types, str):
            data_types = list(data_types)
        for data_type in data_types:
            if data_type not in ["train", "dev", "test"]:
                raise ValueError("Unknown data_type: {}, only train, dev and test "
                                 "datatypes are allowed".format(data_type))
        if isinstance(data_path, str):
            data_path = Path(data_path)
        if isinstance(data_path, Path):
            if data_path.exists():
                is_file = data_path.is_file()
            else:
                is_file = (len(data_types) == 1)
            if is_file:
                # path to a single file
                data_path, reserve_data_path = [data_path], None
            else:
                # path to data directory
                if language is None:
                    raise ValueError("You must implicitly provide language "
                                     "when providing data directory as source")
                reserve_data_path = data_path
                data_path = [data_path / "{}-ud-{}.conllu".format(language, mode)
                             for mode in data_types]
                reserve_data_path = [
                    reserve_data_path / language / "{}-ud-{}.conllu".format(language, mode)
                    for mode in data_types]
        else:
            data_path = [Path(data_path) for data_path in data_path]
            reserve_data_path = None
        if len(data_path) != len(data_types):
            raise ValueError("The number of input files in data_path and data types "
                             "in data_types must be equal")
        has_missing_files = any(not filepath.exists() for filepath in data_path)
        if has_missing_files and reserve_data_path is not None:
            has_missing_files = any(not filepath.exists() for filepath in reserve_data_path)
            if not has_missing_files:
                data_path = reserve_data_path
        if has_missing_files:
            # Files are downloaded from the Web repository
            dir_path = data_path[0].parent
            language = language or get_language(data_path[0].parts[-1])
            url = self.URL + "{}.tar.gz".format(language)
            log.info('[downloading data from {} to {}]'.format(url, dir_path))
            dir_path.mkdir(exist_ok=True, parents=True)
            download_decompress(url, dir_path)
            mark_done(dir_path)
        data = {}
        for mode, filepath in zip(data_types, data_path):
            if mode == "dev":
                mode = "valid"
#             if mode == "test":
#                 kwargs["read_only_words"] = True
            data[mode] = read_infile(filepath, **kwargs)
            # data[mode] = [((sent, 0), tags) for sent, tags in curr_data]
        return data


@register('morphotagger_multidataset_reader')
class MorphotaggerMultiDatasetReader(DatasetReader):
    """Class to read training datasets in UD format"""

    URL = 'http://files.deeppavlov.ai/datasets/UD2.0_source/'

    def read(self, data_path: Union[List, str],
             additional_data_path: Optional[List] = None,
             language: Optional[str] = "None",
             additional_languages: Optional[List[str]] = None,
             data_types: Optional[List[str]] = None,
             additional_data_types: Optional[Union[str, List[str]]] = None,
             additional_read_params: Optional[Union[List,dict]] = None,
             **kwargs) -> Dict[str, List]:
        """Reads UD dataset from data_path.

        Args:
            data_path: can be either
                1. a directory containing files. The file for data_type 'mode'
                is then data_path / {language}-ud-{mode}.conllu
                2. a list of files, containing the same number of items as data_types
            additional_data_path:
                a list of additional data files (None if no additional data files are provided).
            language: specifies the language of the main dataset in multilingual setting,
                in multiling
            additional_languages: the languages of additional dataset files
            data_types: which dataset parts among 'train', 'dev', 'test' are returned
            additional_data_types: parts of the dataset to which additional files are attached
        Returns:
            a dictionary containing dataset fragments (see ``read_infile``) for given data types
        """
        # main data
        if data_types is None:
            data_types = ["train", "dev"]
        elif isinstance(data_types, str):
            data_types = list(data_types)
        for data_type in data_types:
            if data_type not in ["train", "dev", "test"]:
                raise ValueError("Unknown data_type: {}, only train, dev and test "
                                 "datatypes are allowed".format(data_type))
        if isinstance(data_path, str):
            data_path = Path(data_path)
        if isinstance(data_path, Path):
            if data_path.exists():
                is_file = data_path.is_file()
            else:
                is_file = (len(data_types) == 1)
            if is_file:  # path to a single file
                data_path = [data_path], None
            else:  # path to data directory
                if language == "None":
                    raise ValueError("You must implicitly provide language "
                                     "when providing data directory as source")
                data_path = [data_path / "{}-ud-{}.conllu".format(language, mode)
                             for mode in data_types]
        else:
            data_path = [Path(data_path) for data_path in data_path]
        if len(data_path) != len(data_types):
            raise ValueError("The number of input files in data_path and data types "
                             "in data_types must be equal")
        has_missing_files = any(not filepath.exists() for filepath in data_path)
        if has_missing_files:
            # Files are downloaded from the Web repository
            dir_path = data_path[0].parent
            language = language or get_language(data_path[0].parts[-1])
            url = self.URL + "{}.tar.gz".format(language)
            log.info('[downloading data from {} to {}]'.format(url, dir_path))
            dir_path.mkdir(exist_ok=True, parents=True)
            download_decompress(url, dir_path)
            mark_done(dir_path)
        data = {mode: [] for mode in ["train", "valid", "test"]}
        for mode, filepath in zip(data_types, data_path):
            if mode == "dev":
                mode = "valid"
            curr_data = read_infile(filepath, **kwargs)
            data[mode].extend([((sent, language), tags) for sent, tags in curr_data])
        if additional_data_path is not None:
            if additional_read_params is None:
                additional_read_params = dict()
            if isinstance(additional_read_params, dict):
                additional_read_params = [additional_read_params] * len(additional_data_path)
            if len(additional_read_params) != len(additional_data_path):
                raise ValueError("Additional_read_params should have the same length as additional_data_path")
            if additional_data_types is None:
                additional_data_types = ["train"] * len(additional_data_path)
            if len(additional_data_types) != len(additional_data_path):
                raise ValueError("Additional_data_types should have the same length as additional_data_path")
            if isinstance(additional_languages, str):
                additional_languages = [additional_languages] * len(additional_data_path)
            if len(additional_languages) != len(additional_data_path):
                raise ValueError("There must be the same number of languages in additional_languages"
                                 "as files in additional_data_path")
        else:
            additional_data_path = []
        for i, filepath in enumerate(additional_data_path):
            if isinstance(filepath, str):
                filepath = Path(filepath)
            filepath = expand_path(filepath)
            curr_read_params, curr_language, curr_data_type =\
                additional_read_params[i], additional_languages[i], additional_data_types[i]
            curr_data = read_infile(filepath, **curr_read_params)
            curr_data = [((sent, curr_language), tags) for sent, tags in curr_data]
            data[curr_data_type].extend(curr_data)
        return data
