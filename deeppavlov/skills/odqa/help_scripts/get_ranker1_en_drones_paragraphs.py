"""
Evaluate Ranker1 on SQuAD dataset.

SQuAD dataset for this evaluation script should be JSON-formatted as following:

'{"question": "q1", "answers": ["a10...a1n"]}'
...
'{"question": "qN", "answers": ["aN0...aNn"]}'

"""
import argparse
import time
import unicodedata
import logging
import csv

from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.dataset_iterators.sqlite_iterator import SQLiteDataIterator

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
file = logging.FileHandler('../eval_logs/ranker1_toloka.log')
file.setFormatter(fmt)
logger.addHandler(file)

parser = argparse.ArgumentParser()

parser.add_argument("-config_path", help="path to a JSON ranker config", type=str,
                    default='/media/olga/Data/projects/iPavlov/DeepPavlov/deeppavlov/configs/odqa/en_ranker_infer_drones.json')
parser.add_argument("-dataset_path", help="path to a JSON formatted dataset", type=str,
                    default='/media/olga/Data/projects/ODQA/data/PMEF/QA_heli - v2.csv')
parser.add_argument("-database_url", help="path to a SQLite database with wikipedia articles",
                    type=str,
                    default='http://lnsigo.mipt.ru/export/datasets/pmef/en_drones.db')
parser.add_argument("-n", "--number-retrieve", help="top n documents to retrieve", default=5,
                    type=int)


def encode_utf8(s: str):
    return unicodedata.normalize('NFD', s).encode('utf-8')  # encode_from_strings deprecated


def instance_score(answers, texts):
    for a in answers:
        for doc_text in texts:
            if doc_text.find(a) != -1:
                return 1
    return 0


def read_csv(csv_path):
    output = []
    with open(csv_path) as fin:
        reader = csv.reader(fin)
        next(reader)
        for item in reader:
            output.append({'question': item[0], 'answer': item[1]})
    return output


def main():
    args = parser.parse_args()
    config = read_json(args.config_path)
    ranker = build_model_from_config(config)  # chainer
    dataset = read_csv(args.dataset_path)
    iterator = SQLiteDataIterator(data_url=args.database_url)

    dataset_size = len(dataset)
    logger.info('Dataset size: {}'.format(dataset_size))
    correct_answers = 0
    n_queries = 0
    start_time = time.time()

    try:
        mapping = []
        db_size = len(iterator.doc_ids)
        logger.info("DB size: {}".format(db_size))
        ranker.pipe[0][2].top_n = db_size
        for instance in dataset:
            q = instance['question']
            q = unicodedata.normalize('NFD', q)
            result = ranker([q])
            top_n = result[0][0]
            scores = list(result[0][1])

            # formatted_answers = [encode_utf8(a) for a in [instance['answer']]]
            a = instance['answer']

            top_n_texts = [iterator.get_doc_content(doc_id) for doc_id in top_n]
            # top_n_texts = [encode_utf8(text) for text in top_n_texts]
            # top_n_texts = [text for text in top_n_texts]

            context_score_pairs = tuple(zip(top_n_texts, scores))

            # for item in zip(top_n_texts, scores)


            mapping.append((q,
                            a,
                            context_score_pairs))

        import json
        with open('/media/olga/Data/projects/ODQA/data/PMEF/ranker1_drones_all_paragraphs.json', 'w') as fout:
            json.dump(mapping, fout)

    except Exception as e:
        logger.exception(e)
        logger.info("Completed with exception in {} seconds".format(time.time() - start_time))
        raise


if __name__ == "__main__":
    main()

0