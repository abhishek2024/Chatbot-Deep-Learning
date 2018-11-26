import argparse
import time
import unicodedata
import logging
import csv
from urllib import parse

import requests
import numpy as np
# from scipy.spatial.distance import cosine

from deeppavlov.dataset_iterators.sqlite_iterator import SQLiteDataIterator
from deeppavlov.core.common.file import read_json, save_json
from deeppavlov.core.commands.infer import build_model_from_config

from deeppavlov.models.vectorizers.hashing_tfidf_vectorizer import HashingTfIdfVectorizer
from deeppavlov.models.tokenizers.spacy_tokenizer import StreamSpacyTokenizer


logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
# file = logging.FileHandler('../eval_logs/ololo.log')
# file.setFormatter(fmt)
# logger.addHandler(file)

parser = argparse.ArgumentParser()

parser.add_argument("-config_path", help="path to a JSON ranker config", type=str,
                    default='/media/olga/Data/projects/DeepPavlov/deeppavlov/configs/odqa/en_ranker_tfidf_wiki.json')
parser.add_argument("-dataset_path", help="path to a SQuAD dataset", type=str,
                    default='/media/olga/Data/datasets/squad/preproc/train-v1.1_prep_4ranker.json')
parser.add_argument("-save_path", help="path to save a ready dataset", type=str,
                    default='/media/olga/Data/datasets/squad/tfidf_pop_full_articles.csv')
parser.add_argument("-vectorizer_path", help="path to a tfidf matrix for tfidf vectorizer", type=str,
                    default='/media/olga/Data/projects/DeepPavlov/download/odqa/enwiki_tfidf_matrix.npz')
# parser.add_argument("-db_path", help="path to an iterator db path", type=str,
#                     default='/media/olga/Data/projects/DeepPavlov/download/odqa/enwiki.db')
parser.add_argument("-ranker_answers_path", help="path to saved ranker answers", type=str,
                    default='/media/olga/Data/datasets/squad/ranker_tfidf_answers_full_articles_top_10_train.json')

POPULARITIES_PATH = '/media/olga/Data/datasets/squad/popularities.json'
WIKI_URL = 'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/all-agents/{}/monthly/20171105/20181105'


def normalize(s: str):
    return unicodedata.normalize('NFD', s)


def instance_score(answers, texts):
    formatted_answers = [normalize(a.strip().lower()) for a in answers]
    formatted_texts = [normalize(text.lower()) for text in texts]
    for a in formatted_answers:
        for doc_text in formatted_texts:
            if doc_text.find(a) != -1:
                return 1
    return 0


def get_popularities(titles, popularities_path, url):
    title2popularity = read_json(popularities_path)
    result = {}
    for orig_title in titles:
        try:
            try:
                result[orig_title] = title2popularity[orig_title]
                continue
            except KeyError:
                pass

            title = orig_title.replace(' ', '_')
            title = title.replace("%", "%25")
            # title = title.replace("/", "////")
            title = title.replace('&amp;', '%26')
            title = title.replace('&quot;', '\"')
            title = title.replace("?", "%3F")
            title = title.replace("\'", "%27")
            title = unicodedata.normalize('NFC', title)
            title = title.replace("è:", "è:")
            title = title.replace("ù_", "ù_")
            title = title.replace("ü", "ü")
            title = title.replace("ī", "ī")
            title = title.replace("+", '%2B')

            if orig_title == "&quot;Chūsotsu&quot; &quot;Chūkara&quot;":
                title = '\"Chūsotsu\" \"Chūkara\"'
            elif orig_title == "&quot;The Spaghetti Incident?&quot;":
                title = "The_Spaghetti_Incident%3F"
            elif orig_title == "&quot;Üns&quot; Theatre":
                title = "\"Üns\"_Theatre"
            elif orig_title == ".40 S&amp;W":
                title = ".40_S%26W"
            elif orig_title == "1/2 &amp; 1/2":
                title = "1/2_&_1/2"
            elif orig_title == "1/2 + 1/4 + 1/8 + 1/16 + ⋯":
                title = "1/2 + 1/4 + 1/8 + 1/16 + ⋯"

            f_url = url.format(title)
            response = requests.get(f_url).json()
            try:
                popularities = [item['views'] for item in response['items']]
            except KeyError:
                title = parse.quote_plus(title)
                f_url = WIKI_URL.format(title)
                response = requests.get(f_url).json()
                popularities = [item['views'] for item in response['items']]
            popularity = sum(popularities) / len(popularities)
            result[orig_title] = popularity

        except Exception:
            print(f'Exception in original title {orig_title}')
            raise
    return result


def main():
    tokenizer = StreamSpacyTokenizer(stopwords='sklearn', alphas_only=True, ngram_range=[1, 2], lemmas=True)
    args = parser.parse_args()
    config = read_json(args.config_path)
    ranker = build_model_from_config(config)  # chainer
    dataset = read_json(args.dataset_path)
    dataset = dataset[:10]

    qa_dataset_size = len(dataset)
    logger.info('QA dataset size: {}'.format(qa_dataset_size))
    # n_queries = 0  # DEBUG
    start_time = time.time()
    TEXT_IDX = 1
    SCORE_IDX = 2
    TITLE_IDX = 3
    IDX_IDX = 4

    TOP_N = 10

    try:

        ranker_answers = ranker([i['question'] for i in dataset])
        save_json(ranker_answers, args.ranker_answers_path)
        # del ranker
        # ranker_answers = read_json('args.ranker_answers_path)
        # iterator = SQLiteDataIterator(load_path=args.db_path)
        # iterator = ranker.pipe[1][2]
        title2index = ranker.pipe[1][2].doc2index
        vectorizer = ranker.pipe[0][2].vectorizer
        del ranker

        # vectorizer = HashingTfIdfVectorizer(load_path=args.vectorizer_path, tokenizer=tokenizer)

        # iterator = ranker.pipe[1][2]

        # chunk_indices = [text[CHUNK_IDX] for answer in ranker_answers for text in answer]
        # chunk_titles = [k for i in chunk_indices for k in iterator.doc2index.keys() if iterator.doc2index[k] == i]
        # i2t = iterator.index2title()
        # del iterator
        # chunk_titles = [i2t[i] for i in chunk_indices]
        # popularities = get_popularities(set(chunk_titles), POPULARITIES_PATH, WIKI_URL)
        popularities = read_json(POPULARITIES_PATH)

        returned_db_size = len(ranker_answers[0])
        logger.info("Returned DB size: {}".format(returned_db_size))

        with open(args.save_path, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow(['Article tfidf score', 'Article popularity', 'Article title', 'Label'])  # header

            for qa, ranker_answer in zip(dataset, ranker_answers):
                correct_answers = qa['answers']
                correct_title = qa['title']
                question = qa['question']
                edit_correct_title = correct_title.replace('_', ' ')
                edit_correct_title = parse.unquote_plus(edit_correct_title)
                edit_correct_title = edit_correct_title.replace('%26', '&amp;')
                edit_correct_title = edit_correct_title.replace('\"', '&quot;')

                try:
                    popularity = popularities[edit_correct_title]
                    tfidf_score = \
                    np.dot(vectorizer([question])[0], vectorizer.tfidf_matrix[:, title2index[edit_correct_title]]).data[
                        0]
                except KeyError:
                    continue
                i_score = 1
                writer.writerow([tfidf_score, popularity, edit_correct_title, i_score])

                for n in range(TOP_N):
                    title = ranker_answer[n][TITLE_IDX]
                    if title == edit_correct_title:
                        continue

                    # text = ranker_answer[n][TEXT_IDX]
                    tfidf_score = ranker_answer[n][SCORE_IDX]
                    # chunk_title = i2t[chunk_id]
                    # i_score = instance_score(correct_answers, [text])
                    i_score = 0
                    popularity = popularities[title]

                    # print(f'Chunk context tfidf: {tfidf_score}')
                    # print(f'Chunk popularity: {popularity}')
                    # print(f'Chunk label: {i_score}')
                    print(f'Article title: {title}')
                    print()
                    writer.writerow([tfidf_score, popularity, title, i_score])

        logger.info("Completed successfully in {} seconds.".format(time.time() - start_time))

    except Exception as e:
        logger.exception(e)
        logger.info("Completed with exception in {} seconds".format(time.time() - start_time))
        raise


if __name__ == "__main__":
    main()
