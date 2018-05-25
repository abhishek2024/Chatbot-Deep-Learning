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
import nmslib

from scipy.sparse import csr_matrix, lil_matrix, vstack

from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.dataset_iterators.sqlite_iterator import SQLiteDataIterator

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
file = logging.StreamHandler()
file.setFormatter(fmt)
logger.addHandler(file)

parser = argparse.ArgumentParser()

parser.add_argument("-config_path", help="path to a JSON ranker config", type=str,
                    default='/media/olga/Data/projects/iPavlov/DeepPavlov/deeppavlov/configs/odqa/en_ranker_infer_drones.json')
parser.add_argument("-dataset_path", help="path to a JSON formatted dataset", type=str,
                    default='/media/olga/Data/projects/ODQA/data/PMEF/QA_heli - v2.csv')
parser.add_argument("-database_url", help="path to a SQLite database with wikipedia articles",
                    type=str,
                    default='http://lnsigo.mipt.ru/export/datasets/drones.db')


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

    data_matrix = ranker.pipe[0][2].tfidf_matrix
    data_matrix = data_matrix.transpose()

    # questions = [q['question'] for q in dataset]
    # answers = [q['answer'] for q in dataset]
    #
    # ranker_class = ranker.pipe[0][2].vectorizer
    # q_vectors = []
    # for q in questions:
    #     q_vectors.append(ranker_class([q]))
    # q_vectors = vstack(q_vectors)


    M = 10
    efC = 100

    num_threads = 4
    index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post': 0}
    index = nmslib.init(method='hnsw', space='cosinesimil_sparse',
                        data_type=nmslib.DataType.SPARSE_VECTOR)
    index.addDataPointBatch(data_matrix)
    index.createIndex(index_time_params)

    efS = 100
    query_time_params = {'efSearch': efS}
    index.setQueryTimeParams(query_time_params)

    # db_size = len(iterator.doc_ids)
    sentences_size = 59

    correct_answers = 0

    start_time = time.time()

    try:
        mapping = {}
        logger.info("Dataset size in sentences: {}".format(sentences_size))

        for k in range(1, sentences_size + 1, 2):
            nbrs = index.knnQueryBatch(q_vectors, k=k, num_threads=num_threads)
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
