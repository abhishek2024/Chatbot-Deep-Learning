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
import re
import string

from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.dataset_iterators.sqlite_iterator import SQLiteDataIterator

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
file = logging.FileHandler('../eval_logs/ranker1_toloka_v2.log')
file.setFormatter(fmt)
logger.addHandler(file)

parser = argparse.ArgumentParser()

parser.add_argument("-config_path", help="path to a JSON ranker config", type=str,
                    default='/media/olga/Data/projects/iPavlov/DeepPavlov/deeppavlov/configs/odqa/en_ranker1_train_drones_chunks.json')
parser.add_argument("-dataset_path", help="path to a JSON formatted dataset", type=str,
                    default='/media/olga/Data/projects/ranker_test/ranker_test/drone_questions.txt')
parser.add_argument("-database_url", help="path to a SQLite database with wikipedia articles",
                    type=str,
                    default='/media/olga/Data/projects/iPavlov/DeepPavlov/download/pmef/en_drones_chunks.db')



def encode_utf8(s: str):
    return unicodedata.normalize('NFD', s).encode('utf-8')  # encode_from_strings deprecated


def instance_score(answers, texts):
    regex = re.compile('[%s]' % re.escape(string.punctuation))

    for a in answers:
        a = regex.sub(' ', a)
        a = re.sub(" +", " ", a)
        a = a.replace('\n', ' ')
        for doc_text in texts:
            doc_text = regex.sub(' ', doc_text)
            doc_text = re.sub(" +", " ", doc_text)
            doc_text = doc_text.replace('\n', ' ')
            if doc_text.find(a.strip()) != -1:
                return 1
    return 0


# def read_csv(csv_path):
#     output = []
#     with open(csv_path) as fin:
#         reader = csv.reader(fin)
#         next(reader)
#         for item in reader:
#             output.append({'question': item[0], 'answer': item[1]})
#     return output

def read_csv(csv_path):
    output = []
    with open(csv_path) as fin:
        reader = csv.reader(fin)
        for item in reader:
            try:
                qa = item[0].split(';')
                output.append({'question': qa[0], 'answer': qa[1]})
            except Exception:
                pass
    return output


def main():
    args = parser.parse_args()
    config = read_json(args.config_path)
    ranker = build_model_from_config(config)  # chainer
    dataset = read_csv(args.dataset_path)
    iterator = SQLiteDataIterator(load_path=args.database_url)

    dataset_size = len(dataset)
    logger.info('Dataset size: {}'.format(dataset_size))
    correct_answers = 0
    n_queries = 0
    start_time = time.time()

    try:
        mapping = {}
        db_size = len(iterator.doc_ids)
        logger.info("DB size: {}".format(db_size))
        for n in range(1, db_size + 1):
            ranker.pipe[0][2].top_n = n
            for instance in dataset:
                q = instance['question']
                q = unicodedata.normalize('NFD', q)
                result = ranker([q])
                top_n = result[0][:n]
                # scores = result[1]

                # formatted_answers = [encode_utf8(a) for a in [instance['answer']]]
                formatted_answers = [a.strip().lower() for a in [instance['answer']]]

                top_n_texts = [iterator.get_doc_content(doc_id) for doc_id in top_n]
                # top_n_texts = [encode_utf8(text) for text in top_n_texts]
                top_n_texts = [text.lower() for text in top_n_texts]

                # TODO check everything is good with encodings (should encode_from_strings
                # from the very beginning?)

                correct_answers += instance_score(formatted_answers, top_n_texts)
                # logger.info('Processed {} queries from {}'.format(n_queries, dataset_size))
                # logger.info('Correct answers: {}'.format(formatted_answers))
                # n_queries += 1
                # logger.info('Current score: {}'.format(correct_answers / n_queries))

            total_correct = correct_answers / dataset_size
            logger.info(
                'Recall for top {}: {}'.format(n, total_correct))
            mapping[n] = total_correct
            correct_answers = 0
        logger.info("Completed successfully in {} seconds.".format(time.time() - start_time))
        logger.info("Quality mapping: {}".format(mapping))

        import matplotlib.pylab as plt

        lists = sorted(mapping.items())  # sorted by key, return a list of tuples

        x, y = zip(*lists)  # unpack a list of pairs into two tuples

        plt.plot(x, y)
        plt.xlabel('quantity of dataset returned by ranker 1')
        plt.ylabel('probability of true answer occurrence')
        plt.show()

    except Exception as e:
        logger.exception(e)
        logger.info("Completed with exception in {} seconds".format(time.time() - start_time))
        raise


if __name__ == "__main__":
    main()
