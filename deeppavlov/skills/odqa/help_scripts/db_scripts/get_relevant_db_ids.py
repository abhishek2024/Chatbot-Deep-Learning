import argparse
import sqlite3
from typing import List, Union
from itertools import chain
from operator import itemgetter

from nltk.tokenize import sent_tokenize

from deeppavlov.core.common.file import read_json, save_json
from deeppavlov.dataset_iterators.sqlite_iterator import SQLiteDataIterator
from deeppavlov.core.commands.infer import build_model_from_config

parser = argparse.ArgumentParser()

parser.add_argument("-ranker_config_path", help="path to a SQuAD dataset preprocessed for ranker", type=str,
                    default="/media/olga/Data/projects/DeepPavlov/deeppavlov/configs/odqa/en_ranker_tfidf_wiki.json")
parser.add_argument("-dataset_path", help="path to a SQuAD dataset preprocessed for ranker", type=str,
                    default="/media/olga/Data/datasets/squad/preproc/train-v1.1_prep_4ranker.json")


def main():
    args = parser.parse_args()
    ranker = build_model_from_config(read_json(args.ranker_config_path))
    dataset = read_json(args.dataset_path)
    # dataset = dataset[:10]

    relevant_ids = []

    for instance in dataset:
        question = instance['question']
        response = ranker([question])
        relevant_ids += [item[3] for item in response[0]]

    save_json(list(set(relevant_ids)), 'relevant_ids.json')


if __name__ == "__main__":
    main()
