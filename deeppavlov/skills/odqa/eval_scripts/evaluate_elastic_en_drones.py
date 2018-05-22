import json
import configparser
import argparse
import time
import unicodedata
import logging
import csv
from operator import itemgetter
import random

import requests
from requests.auth import HTTPBasicAuth
from urllib.parse import urljoin
import elasticsearch

from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.dataset_iterators.sqlite_iterator import SQLiteDataIterator

__all__ = ["ES_AUTH", "ES_HOST", "Elasticsearch"]

config = configparser.ConfigParser()
config.read('elastic.cfg')
ES_HOST = config['DEFAULT']['ESHost']
if 'ESUser' in config['DEFAULT']:
    auth = (config['DEFAULT']['ESUser'], config['DEFAULT']['ESPassword'])
    ES_AUTH = HTTPBasicAuth(*auth)
else:
    auth = None
    ES_AUTH = None


IDX_NAME = 'drones'
_HEAD = {'Content-Type': 'application/json'}
_TYPE = 'text'

_SETTINGS = {
    "settings": {
        "analysis": {
            "filter": {
                "english_stop": {
                    "type": "stop",
                    "stopwords": "_english_"
                },
                "light_english_stemmer": {
                    "type": "stemmer",
                    "language": "light_english"
                },
                "autocomplete_filter": {
                    "type": "edge_ngram",
                    "min_gram": 1,
                    "max_gram": 2
                }
            },
            "analyzer": {
                "custom": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "english_stop",
                        "light_english_stemmer"
                    ]
                }
            }
        }
    },
    "mappings": {
        "text": {
            "properties": {
                "text": {
                    "type": "text",
                    "similarity": "BM25",
                    "analyzer": "custom"
                }
            }
        }
    }
}


def Elasticsearch(url=None, timeout=1000, http_auth=auth):
    if url is None:
        url = ES_HOST
    return elasticsearch.Elasticsearch(url, timeout=timeout, http_auth=http_auth)


def create_index(data):
    for paragraph in data:
        idx = paragraph['title']
        path = "{}/{}/{}".format(IDX_NAME, _TYPE, idx)
        fullPath = urljoin(ES_HOST, path)
        resp = requests.post(fullPath, data=json.dumps(paragraph), headers=_HEAD, auth=ES_AUTH)
        print(resp.status_code)
        print(resp.text)


def create_mapping():
    path = IDX_NAME
    fullPath = urljoin(ES_HOST, path)
    resp = requests.put(fullPath, data=json.dumps(_SETTINGS), headers=_HEAD, auth=ES_AUTH)
    print(resp.status_code)
    print(resp.text)


def delete_index():
    path = IDX_NAME
    fullPath = urljoin(ES_HOST, path)
    resp = requests.delete(fullPath, auth=ES_AUTH)
    resp = requests.delete(urljoin(ES_HOST, '.ltrstore'), auth=ES_AUTH)


def run_elastic(json_path):
    es = Elasticsearch(timeout=30)
    drones_data = read_json(json_path)
    delete_index()
    create_mapping()
    create_index(drones_data)


logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
file = logging.FileHandler('../eval_logs/elastic_toloka_v2.log')
file.setFormatter(fmt)
logger.addHandler(file)

parser = argparse.ArgumentParser()

parser.add_argument("-dataset_path", help="path to a JSON formatted dataset", type=str,
                    default='/media/olga/Data/projects/ODQA/data/PMEF/QA_heli - v2.csv')
parser.add_argument("-database_url", help="path to a SQLite database with wikipedia articles",
                    type=str,
                    default='http://lnsigo.mipt.ru/export/datasets/pmef/en_drones.db')
parser.add_argument("-json_path", help="original data as json", type=str, default="/media/olga/Data/projects/ODQA/data/PMEF/en/001/drones.json")
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

    json_data_path = args.json_path
    run_elastic(json_data_path)

    dataset = read_csv(args.dataset_path)
    iterator = SQLiteDataIterator(data_url=args.database_url)

    dataset_size = len(dataset)
    logger.info('Dataset size: {}'.format(dataset_size))
    correct_answers = 0
    n_queries = 0
    start_time = time.time()

    try:
        mapping = {}
        db_size = len(iterator.doc_ids)
        logger.info("DB size: {}".format(db_size))
        for n in range(1, db_size + 1, 2):
            for instance in dataset:
                q = instance['question']
                q = unicodedata.normalize('NFD', q)

                query = {"query": {"query_string": {"query": q}}}
                path = "_search"
                fullPath = urljoin(ES_HOST, path)
                response = requests.post(fullPath, data=json.dumps(query), headers=_HEAD, auth=ES_AUTH)

                results = []
                try:
                    for hit in json.loads(response.content)['hits']['hits']:
                        text = hit['_source']['text']
                        id = hit['_source']['title']
                        score = hit['_score']
                        results.append((id, score, text))
                except KeyError:
                    print(json.loads(response.content))

                results = sorted(results, key=itemgetter(1))[::-1]

                top_n = [item[0] for item in results[:n]]

                if len(results) < n:
                    shortage = n - len(results)
                    results_ids = [item[0] for item in results]
                    db_ids = iterator.doc_ids
                    lookup_ids = [idx for idx in db_ids if idx not in results_ids]
                    final_ids = random.sample(lookup_ids, shortage)
                    top_n += final_ids

                # formatted_answers = [encode_utf8(a) for a in [instance['answer']]]
                formatted_answers = [a.strip().lower() for a in [instance['answer']]]

                top_n_texts = [iterator.get_doc_content(doc_id) for doc_id in top_n]
                # top_n_texts = [encode_utf8(text) for text in top_n_texts]
                top_n_texts = [text.lower() for text in top_n_texts]

                # TODO check everything is good with encodings (should encode_from_strings
                # from the very beginning?)

                correct_answers += instance_score(formatted_answers, top_n_texts)
                logger.info('Processed {} queries from {}'.format(n_queries, dataset_size))
                # logger.info('Correct answers: {}'.format(formatted_answers))
                n_queries += 1
                # logger.info('Current score: {}'.format(correct_answers / n_queries))
            n_queries = 0

            total_correct = correct_answers / dataset_size
            logger.info(
                'Percentage of the instances for which the correct document was'
                ' retrieved in top {} retrieved documents: {}'.format(n, total_correct))
            mapping[n / db_size] = total_correct
            correct_answers = 0
        logger.info("Completed successfully in {} seconds.".format(time.time() - start_time))
        logger.info("Quality mapping: {}".format(mapping))

        import matplotlib.pylab as plt

        lists = sorted(mapping.items())  # sorted by key, return a list of tuples

        x, y = zip(*lists)  # unpack a list of pairs into two tuples

        plt.plot(x, y)
        plt.xlabel('quantity of dataset returned by elastic')
        plt.ylabel('probability of true answer occurrence')
        plt.show()

    except Exception as e:
        logger.exception(e)
        logger.info("Completed with exception in {} seconds".format(time.time() - start_time))
        raise


if __name__ == "__main__":
    main()

