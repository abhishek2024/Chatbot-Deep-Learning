import argparse
import sqlite3
from typing import List, Union
from itertools import chain

from nltk.tokenize import sent_tokenize

from deeppavlov.core.common.file import read_json
from deeppavlov.dataset_iterators.sqlite_iterator import SQLiteDataIterator

parser = argparse.ArgumentParser()

parser.add_argument("-input_db_path",
                    help="path to an input db with wikipedia articles \
                    (BETTER MAKE A COPY SINCE IT WILL BE CHANGED IN RUNTIME)",
                    type=str,
                    default="/media/olga/Data/projects/DeepPavlov/download/odqa/enwiki_chunk.db")
parser.add_argument("-new_db_path", help="path to a SQuAD dataset preprocessed for ranker", type=str,
                    default="/media/olga/Data/projects/DeepPavlov/download/odqa/enwiki_chunk.db")


def main():
    args = parser.parse_args()


if __name__ == "__main__":
    main()
