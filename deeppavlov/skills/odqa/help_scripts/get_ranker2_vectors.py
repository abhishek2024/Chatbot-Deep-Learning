import argparse
import json
import pickle
import unicodedata
import csv
from io import StringIO

import numpy

from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.infer import build_model_from_config

parser = argparse.ArgumentParser()

parser.add_argument("-config_path", help="path to a JSON ranker config", type=str,
                    default='../../../../deeppavlov/configs/odqa/en_ranker2_infer_drones.json')
parser.add_argument("-dataset_path", help="path to a JSON formatted dataset", type=str,
                    default='/home/gureenkova/data/QA_heli - v2.csv')
parser.add_argument("-output_path", help="path to a an output JSON", type=str,
                    default='/home/gureenkova/data/ranker2_vectors.json')


def encode_utf8(s: str):
    return unicodedata.normalize('NFD', s).encode('utf-8')  # encode_from_strings deprecated


def read_csv(csv_path):
    output = []
    with open(csv_path) as fin:
        reader = csv.reader(fin)
        next(reader)
        for row in reader:
            output.append({'question': row[0], 'answer': row[1]})
    return output


def main():
    args = parser.parse_args()
    config = read_json(args.config_path)
    ranker = build_model_from_config(config)  # chainer
    dataset = read_csv(args.dataset_path)
    output_path = args.output_path

    questions = [item['question'] for item in dataset]

    with open(output_path, 'wb') as fout:
        result = ranker(questions)
        pickle.dump(result, fout, protocol=0)  # protocol 0 is printable ASCII
        print('Saved!')
        # deserialized_a = pickle.loads(serialized)


if __name__ == "__main__":
    main()
