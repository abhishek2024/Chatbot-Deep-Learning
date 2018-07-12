"""
Count a ranker recall.
Need:
1. A ranker config (with rank, text, score, text_id API) for a specific domain (eg. "en_drones")
2. QA dataset for this domain.
"""

import argparse
import time
import unicodedata
import logging
import csv
import re
import string
import matplotlib.pylab as plt

from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.core.common.file import read_json
from deeppavlov.models.tokenizers.spacy_tokenizer import StreamSpacyTokenizer

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
file = logging.FileHandler('../eval_logs/ranker_ensemble_en_wiki.log')
file.setFormatter(fmt)
logger.addHandler(file)

parser = argparse.ArgumentParser()

parser.add_argument("-config_path", help="path to a JSON ranker config", type=str,
                    default='../../../../deeppavlov/configs/odqa/en_ranker_tfidf_wiki.json')
parser.add_argument("-dataset_path", help="path to SQuAD dataset", type=str,
                    default='/media/olga/Data/projects/ODQA/data/squad/preproc/dev-v1.1_prep_4ranker.json')

TOKENIZER = StreamSpacyTokenizer()


def normalize(s: str):
    return unicodedata.normalize('NFD', s)


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException:
        return False
    return pattern.search(text) is not None


def pmef_instance_score(answers, texts):
    formatted_answers = [normalize(a.strip().lower()) for a in answers]
    formatted_texts = [normalize(text.lower()) for text in texts]
    regex = re.compile('[%s]' % re.escape(string.punctuation))

    for a in formatted_answers:
        a = regex.sub(' ', a)
        a = re.sub(" +", " ", a)
        a = a.replace('\n', ' ')
        for doc_text in formatted_texts:
            doc_text = regex.sub(' ', doc_text)
            doc_text = re.sub(" +", " ", doc_text)
            doc_text = doc_text.replace('\n', ' ')
            if doc_text.find(a.strip()) != -1:
                return 1
    return 0


def instance_score(answers, texts):
    formatted_answers = [normalize(a.strip().lower()) for a in answers]
    formatted_texts = [normalize(text.lower()) for text in texts]
    for a in formatted_answers:
        for doc_text in formatted_texts:
            if doc_text.find(a) != -1:
                return 1
    return 0


def drqa_instance_score(answers, texts, match='string'):
    """Check if a document contains an answer string.

        If `match` is string, token matching is done between the text and answer.
        If `match` is regex, we search the whole text with the regex.
        """
    text = ' '.join([normalize(text.lower()) for text in texts])
    if match == 'string':
        # Answer is a list of possible strings
        text = TOKENIZER(text)
        for single_answer in answers:
            single_answer = normalize(single_answer)
            single_answer = TOKENIZER(single_answer)
            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i: i + len(single_answer)]:
                    return True
    elif match == 'regex':
        # Answer is a regex
        single_answer = normalize(answers[0])
        if regex_match(text, single_answer):
            return True
    return False


def read_csv(csv_path):
    output = []
    with open(csv_path) as fin:
        reader = csv.reader(fin)
        for item in reader:
            try:
                qa = item[0].split(';')
                output.append({'question': qa[0], 'answer': qa[1]})
            except Exception:
                logger.info("Exception in read_csv()")
    return output


def main():
    args = parser.parse_args()
    config = read_json(args.config_path)
    ranker = build_model_from_config(config)  # chainer
    dataset = read_json(args.dataset_path)
    # dataset = dataset[:10]

    qa_dataset_size = len(dataset)
    logger.info('QA dataset size: {}'.format(qa_dataset_size))
    # n_queries = 0  # DEBUG
    start_time = time.time()
    TEXT_IDX = 1

    try:
        mapping = {}
        pmef_mapping = {}
        drqa_mapping = {}

        ranker_answers = ranker([i['question'] for i in dataset])
        returned_db_size = len(ranker_answers[0])
        logger.info("Returned DB size: {}".format(returned_db_size))

        for n in range(1, 11):
            correct_answers = 0
            pmef_correct_answers = 0
            drqa_correct_answers = 0
            i = 1
            for qa, ranker_answer in zip(dataset, ranker_answers):
                answers = qa['answers']
                texts = [answer[TEXT_IDX] for answer in ranker_answer[:n]]
                correct_answers += instance_score(answers, texts)
                pmef_correct_answers += pmef_instance_score(answers, texts)
                drqa_correct_answers += drqa_instance_score(answers, texts)
                logger.info(
                    "Processed {} qa pairs for n={}".format(i, n))
                i += 1
            total_score_on_top_i = correct_answers / qa_dataset_size
            pmef_total_score_on_top_i = pmef_correct_answers / qa_dataset_size
            drqa_total_score_on_top_i = drqa_correct_answers / qa_dataset_size

            logger.info(
                'Recall for top {}: {}'.format(n, total_score_on_top_i))
            logger.info(
                'PMEF recall for top {}: {}'.format(n, pmef_total_score_on_top_i))
            logger.info(
                'DRQA recall for top {}: {}'.format(n, pmef_total_score_on_top_i))
            mapping[n] = total_score_on_top_i
            pmef_mapping[n] = pmef_total_score_on_top_i
            drqa_mapping[n] = drqa_total_score_on_top_i

        logger.info("Completed successfully in {} seconds.".format(time.time() - start_time))
        logger.info("Quality mapping: {}".format(mapping))
        logger.info("PMEF quality mapping: {}".format(pmef_mapping))
        logger.info("DRQA quality mapping: {}".format(drqa_mapping))

        # Plot quality mapping

        lists = sorted(mapping.items())

        x, y = zip(*lists)
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
