"""
Evaluate Ranker1 on SQuAD dataset.

SQuAD dataset for this evaluation script should be JSON-formatted as following:

'{"question": "q1", "answers": ["a10...a1n"]}'
...
'{"question": "qN", "answers": ["aN0...aNn"]}'

"""
import argparse
import json
import unicodedata
import csv

from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.infer import build_model_from_config


parser = argparse.ArgumentParser()

parser.add_argument("-config_path", help="path to a JSON ranker config", type=str,
                    default='../../../../deeppavlov/configs/odqa/en_ranker2_infer_drones.json')
parser.add_argument("-dataset_path", help="path to a JSON formatted dataset", type=str,
                    default='/media/olga/Data/projects/ODQA/data/PMEF/qa_to_photo_toloka_revised.csv')
parser.add_argument("-output_path", help="path to a an output JSON", type=str,
                    default='/media/olga/Data/projects/ODQA/data/PMEF/qa_to_photo_toloka_revised_ranker2_output.json')


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

    with open(output_path, 'w') as fout:
        result = []
        i = 0
        for instance in dataset:
            print('Processing {} query'.format(i))
            q = instance['question']
            a = instance['answer']
            pred = ranker([q])
            result.append({'question': q, 'answer': a, 'context': pred})
            i += 1
        json.dump(result, fout)


if __name__ == "__main__":
    main()
