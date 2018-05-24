"""
Evaluate ranker1+ranker2+SQUAD combination on drones toloka dataset.

"""
import argparse
import time
import unicodedata
import logging
import sys
from pathlib import Path
import csv
import pickle

import sys
import nmslib
import time
import math
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import numpy

root_path = (Path(__file__) / ".." / ".." / ".." / ".." / "..").resolve()
sys.path.append(str(root_path))

from deeppavlov.dataset_iterators.sqlite_iterator import SQLiteDataIterator

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
file = logging.FileHandler('../eval_logs/hnsw_en_drones.log')
file.setFormatter(fmt)
logger.addHandler(file)

parser = argparse.ArgumentParser()

parser.add_argument("-dataset_path", help="path to a JSON formatted dataset", type=str,
                    default='/media/olga/Data/projects/ODQA/data/PMEF/QA_heli - v2.csv')
parser.add_argument("-sentences_path", help="path to sentences from the database",
                    type=str,
                    default='/media/olga/Data/projects/ODQA/data/PMEF/en_drones_sentences.txt')
parser.add_argument("-vectors_path", help="path to a JSON formatted dataset", type=str,
                    default='/media/olga/Data/projects/ODQA/data/PMEF/ranker2_vectors.json')


def encode_utf8(s: str):
    return unicodedata.normalize('NFD', s).encode_from_strings('utf-8')


def read_csv(csv_path):
    output = []
    with open(csv_path) as fin:
        reader = csv.reader(fin)
        next(reader)
        for item in reader:
            output.append({'question': item[0], 'answer': item[1]})
    return output


def unpickle(fpath):
    with open(fpath, 'rb') as fin:
        return pickle.load(fin)


def instance_score(answers, texts):
    for a in answers:
        for doc_text in texts:
            if doc_text.find(a) != -1:
                return 1
    return 0


def main():
    args = parser.parse_args()

    vectors = unpickle(args.vectors_path)
    dataset = read_csv(args.dataset_path)
    # iterator = SQLiteDataIterator(data_url=args.database_url)

    questions = [item['question'] for item in dataset]
    answers = [item['answer'] for item in dataset]

    data_matrix = vectors[0][1]
    query_matrix = numpy.array([item[0] for item in vectors]).squeeze()

    M = 15
    efC = 100
    num_threads = 4
    space_name = 'l2'
    index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC}
    index = nmslib.init(method='hnsw', space=space_name, data_type=nmslib.DataType.DENSE_VECTOR)
    index.addDataPointBatch(data_matrix)
    index.createIndex(index_time_params)

    efS = 100
    query_time_params = {'efSearch': efS}
    index.setQueryTimeParams(query_time_params)

    # db_size = len(iterator.doc_ids)
    sentences_size = len(vectors[0][1])

    correct_answers = 0

    start_time = time.time()

    try:
        mapping = {}
        logger.info("Dataset size in sentences: {}".format(sentences_size))

        for k in range(1, sentences_size + 1, 2):
            nbrs = index.knnQueryBatch(query_matrix, k=k, num_threads=num_threads)
            # formatted_answers = [encode_utf8(a) for a in [instance['answer']]]
            for i, instance in enumerate(dataset):

                nbr = nbrs[i]
                # q = questions[i]
                a = answers[i]

                ids = list(nbr[0])
                # contents = [iterator.get_doc_content(idx) for idx in ids]

                with open(args.sentences_path, 'r') as fin:
                    contents = fin.read().split('\n')

                top_n = []
                for i in range(len(contents)):
                    if i in ids:
                        top_n.append(contents[i])

                formatted_answers = [a.strip().lower()]

                top_n_texts = [text.lower() for text in top_n]

                # TODO check everything is good with encodings (should encode_from_strings
                # from the very beginning?)

                correct_answers += instance_score(formatted_answers, top_n_texts)
                # logger.info('Processed {} queries from {}'.format(n_queries, dataset_size))
                # logger.info('Correct answers: {}'.format(formatted_answers))
                # n_queries += 1
                # logger.info('Current score: {}'.format(correct_answers / n_queries))

            total_correct = correct_answers / len(dataset)
            logger.info(
                'Percentage of the instances for which the correct document was'
                ' retrieved in top {} retrieved documents: {}'.format(k, total_correct))
            mapping[k / sentences_size] = total_correct
            correct_answers = 0
        logger.info("Completed successfully in {} seconds.".format(time.time() - start_time))
        logger.info("Quality mapping: {}".format(mapping))

        import matplotlib.pylab as plt

        lists = sorted(mapping.items())  # sorted by key, return a list of tuples

        x, y = zip(*lists)  # unpack a list of pairs into two tuples

        plt.plot(x, y)
        plt.xlabel('quantity of dataset returned by hnsw(nslib)')
        plt.ylabel('probability of true answer occurrence')
        plt.show()

    except Exception as e:
        logger.exception(e)
        logger.info("Completed with exception in {} seconds".format(time.time() - start_time))
        raise


if __name__ == "__main__":
    main()
