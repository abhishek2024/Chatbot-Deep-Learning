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
import json
import logging
from pathlib import Path
import unicodedata
import sqlite3
from typing import Union, List, Tuple, Generator, Any

from overrides import overrides
from multiprocessing import Pool as ProcessPool
from tqdm import tqdm

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.commands.utils import expand_path

logger = logging.getLogger(__name__)


@register('odqa_reader')
class ODQADataReader(DatasetReader):

    def __init__(self, save_path: str, dataset_format='txt', **kwargs):
        """
        :param save_path: a path to an output SQLite DB
        :param dataset_format: format of dataset files, choose from {'txt', 'json', 'wiki', 'sqlite'}
        """
        self.save_path = save_path
        self.dataset_format = dataset_format

    @overrides
    def read(self, data_path: Union[Path, str], *args, **kwargs) -> None:
        """
        :param data_path: a folder with wikiextractor-formatted files
        :return: None
        """
        # DEBUG
        # if Path(self.save_path).exists():
        #     Path(self.save_path).unlink()

        logger.info('Reading files...')
        if self.dataset_format == 'sqlite':
            return
        self._build_db(data_path)

        # DEBUG
        # if Path(self.save_path).exists():
        #     Path(self.save_path).unlink()

    def iter_files(self, path: Union[Path, str]) -> Generator[Path, Any, Any]:
        path = Path(path)
        if path.is_file():
            yield path
        elif path.is_dir():
            for item in path.iterdir():
                yield self.iter_files(item)
        else:
            raise RuntimeError("Path doesn't exist: {}".format(path))

    def _build_db(self, data_path: Union[Path, str], num_workers=1):
        logger.info('Building the database...')
        conn = sqlite3.connect(str(expand_path(self.save_path)))
        c = conn.cursor()
        sql_table = "CREATE TABLE documents (id PRIMARY KEY, text);"
        c.execute(sql_table)

        files = [f for f in self.iter_files(data_path)]
        workers = ProcessPool(num_workers)
        with tqdm(total=len(files)) as pbar:
            if self.dataset_format == 'txt':
                fn = self._get_file_contents
            elif self.dataset_format == 'wiki':
                fn = self._get_wiki_contents
            elif self.dataset_format == 'json':
                fn = self._get_json_contents
            else:
                raise RuntimeError('Unknown dataset format.')
            for data in tqdm(workers.imap_unordered(fn, files)):
                c.executemany("INSERT INTO documents VALUES (?,?)", data)
                pbar.update()
        conn.commit()
        conn.close()

    def _get_json_contents(self) -> List[Tuple[str, str]]:
        """
        Read a single json file.
        :return: tuple of file names and contents
        """

        docs = []
        with open(self.save_path) as fin:
            for line in fin:
                data = json.loads(line)
                for doc in data:
                    if not doc:
                        continue
                    title = doc['title']
                    normalized_title = unicodedata.normalize('NFD', title)
                    text = doc['text']
                    normalized_text = unicodedata.normalize('NFD', text)
                    docs.append((normalized_title, normalized_text))
        return docs

    @staticmethod
    def _get_wiki_contents(path) -> List[Tuple[str, str]]:
        """
        Read a single wikipedia-extractor formatted file.
        :param path: path to a wikipedia-formatted file
        :return: tuple of file names and contents
        """
        docs = []
        with open(path) as fin:
            for line in fin:
                doc = json.loads(line)
                if not doc:
                    continue
                title = doc['title']
                normalized_title = unicodedata.normalize('NFD', title)
                text = doc['text']
                normalized_text = unicodedata.normalize('NFD', text)
                docs.append((normalized_title, normalized_text))
        return docs

    @staticmethod
    def _get_file_contents(path) -> List[Tuple[str, str]]:
        """
        Read a single txt file.
        :param path: path to a txt file
        :return: tuple of file names and contents
        """
        with open(path) as fin:
            text = fin.read()
            normalized_title = unicodedata.normalize('NFD', path.name)
            normalized_text = unicodedata.normalize('NFD', text)
            return [(normalized_title, normalized_text)]

