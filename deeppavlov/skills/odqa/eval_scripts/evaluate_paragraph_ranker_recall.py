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

from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.infer import build_model_from_config

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
file = logging.FileHandler('../eval_logs/ranker2_squad_dev_ru.log')
file.setFormatter(fmt)
logger.addHandler(file)

parser = argparse.ArgumentParser()

parser.add_argument("-config_path", help="path to a JSON ranker config", type=str,
                    default='../../../../deeppavlov/configs/odqa/ru_ranker_infer_exp_paragraph_ranker.json')
parser.add_argument("-dataset_path", help="path to a JSON formatted dataset", type=str,
                    default='/media/olga/Data/projects/ODQA/data/ru_squad/preproc/dev-v1.1_prep_4ranker.json')
parser.add_argument("-database_url", help="path to a SQLite database with wikipedia articles",
                    type=str,
                    default='http://lnsigo.mipt.ru/export/datasets/wikipedia/ruwiki.db')


def encode_utf8(s: str):
    return unicodedata.normalize('NFD', s).encode('utf-8')  # encode_from_strings deprecated


def instance_score(answers, texts):
    for a in answers:
        for doc_text in texts:
            if doc_text.find(a) != -1:
                return 1
    return 0


def main():
    args = parser.parse_args()
    config = read_json(args.config_path)
    ranker = build_model_from_config(config)  # chainer
    dataset = read_json(args.dataset_path)

    dataset_size = len(dataset)
    logger.info('Dataset size: {}'.format(dataset_size))
    correct_answers = 0
    n_queries = 0
    start_time = time.time()

    try:
        for instance in dataset:
            q = instance['question']
            q = unicodedata.normalize('NFD', q)
            paragraphs = ranker([q])

            answers = instance['answers']
            formatted_answers = [encode_utf8(a) for a in answers]
            top_n_texts = [encode_utf8(text) for text in paragraphs]

            correct_answers += instance_score(formatted_answers, top_n_texts)
            logger.info('Processed {} queries from {}'.format(n_queries, dataset_size))
            logger.info('Correct answers: {}'.format(answers))
            logger.info('Correct answers bytes: {}'.format(formatted_answers))
            n_queries += 1
            logger.info('Current score: {}'.format(correct_answers / n_queries))

        total_correct = correct_answers / dataset_size
        logger.info(
            'Percentage of the instances for which the correct document was'
            ' retrieved in retrieved documents: {}'.format(total_correct))
        logger.info("Completed successfully in {} seconds.".format(time.time() - start_time))
    except Exception as e:
        logger.exception(e)
        logger.info("Completed with exception in {} seconds".format(time.time() - start_time))
        raise


if __name__ == "__main__":
    main()
