"""
Find and output all Wikipedia articles from SQuAD dataset
"""

import argparse
import unicodedata
from typing import List, Dict, Union
import json

from deeppavlov.dataset_iterators.sqlite_iterator import SQLiteDataIterator

parser = argparse.ArgumentParser()

parser.add_argument("squad_data_path",
                    help="path to a SQuAD dataset prepared for ranker", type=str)
parser.add_argument("db_path", help="local path or URL to a SQLite DB with Wikipedia articles",
                    type=str)
parser.add_argument("output_path", help="path to an output JSON file", type=str)


def encode_utf8(s: str):
    return unicodedata.normalize('NFD', s).encode('utf-8')


def read_json(fpath):
    with open(fpath) as fin:
        return json.load(fin)


def write_json(data, fpath):
    with open(fpath, 'w') as fout:
        json.dump(data, fout)


def main():
    args = parser.parse_args()

    iterator = SQLiteDataIterator(load_path=args.db_path)
    dataset = read_json(args.squad_data_path)

    db_ids = set(iterator.get_doc_ids())
    dataset_ids = set([instance['title'] for instance in dataset])
    dataset_ids = {title.replace('_', ' ') for title in dataset_ids}

    intersect_ids = db_ids.intersection(dataset_ids)
    diff_ids = dataset_ids.difference(intersect_ids)
    try:
        assert len(intersect_ids) == len(dataset_ids)
    except AssertionError:
        print('\nTitles that do not exist in the database: {}'.format(diff_ids))
        print()
        pass

    result = []
    for title in intersect_ids:
        result.append({'title': title.replace(' ', '_'), 'text': iterator.get_doc_content(title)})
    for title in diff_ids:
        result.append({'title': title.replace(' ', '_'), 'text': ""})

    write_json(result, args.output_path)

    print('Done!')


if __name__ == "__main__":
    main()
