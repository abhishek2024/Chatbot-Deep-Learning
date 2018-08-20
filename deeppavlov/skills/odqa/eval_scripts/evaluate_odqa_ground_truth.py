"""
Count ODQA f1 choosing answer with max logit returned by squad model.
Need:
1. A ranker config (with rank, text, score, text_id API) for a specific domain (eg. "en_drones")
2. SQuAD dataset.
3. Squad reader config (should return logits).
"""

import argparse
import time
import logging
from operator import itemgetter
import sys
from pathlib import Path

root_path = (Path(__file__) / ".." / ".." / ".." / ".." / "..").resolve()
sys.path.append(str(root_path))

from nltk.tokenize import sent_tokenize

from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.core.common.file import read_json
from deeppavlov.metrics.squad_metrics import squad_f1
from deeppavlov.skills.odqa.help_scripts.split_paragraphs_squad_limit import make_chunks

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
file = logging.FileHandler('../eval_logs/odqa_f1_logits_groud_truth.log')
file.setFormatter(fmt)
logger.addHandler(file)

parser = argparse.ArgumentParser()

parser.add_argument("-dataset_path", help="path to SQuAD dataset", type=str,
                    default='/media/olga/Data/datasets/squad/preproc/dev-v1.1_prep_4odqa.json')
parser.add_argument("-reader_config_path", help="path to squad reader config", type=str,
                    default='../../../../deeppavlov/configs/squad/squad.json')
parser.add_argument("-ground_truth_articles_path", help="path to squad ground truth articles", type=str,
                    default='/media/olga/Data/datasets/squad/wiki_articles/squad_dev_articles.json')


def main():
    args = parser.parse_args()
    reader_config = read_json(args.reader_config_path)
    reader = build_model_from_config(reader_config)  # chainer
    # print(reader([("Deep Pavlov killed Kenny.", "Who killed Kenny?")]))
    dataset = read_json(args.dataset_path)
    # dataset = dataset[:10]

    qa_dataset_size = len(dataset)
    logger.info('QA dataset size: {}'.format(qa_dataset_size))

    ground_truth_texts = read_json(args.ground_truth_articles_path)
    # n_queries = 0  # DEBUG
    start_time = time.time()
    SQUAD_LIMIT = 4000
    SQUAD_BATCH_SIZE = 2

    try:

        y_true_text = [instance['answers'] for instance in dataset]
        y_true_start = [instance['answers_start'] for instance in dataset]
        y_true = list(zip(y_true_text, y_true_start))

        all_y_pred = []
        i = 0

        for instance in dataset:
            title = instance['title']
            question = instance['question']

            ground_truth_text = [d['text'] for d in ground_truth_texts if d['title'] == title][0]

            sentences = sent_tokenize(ground_truth_text)
            chunks = make_chunks(sentences, SQUAD_LIMIT)

            questions = [question] * len(chunks)

            reader_feed = list(zip(chunks, questions))
            batches = [reader_feed[x:x + SQUAD_BATCH_SIZE] for x in range(0, len(reader_feed), SQUAD_BATCH_SIZE)]
            y_pred = []
            for batch in batches:
                y_pred += reader(batch)

            best_y_pred = sorted(y_pred, key=itemgetter(3), reverse=True)[0][:2]
            all_y_pred.append(best_y_pred)
            logger.info(f"Processed {i} instance")
            i += 1

        f1_score = squad_f1(y_true, all_y_pred)
        logger.info("Completed successfully in {} seconds.".format(time.time() - start_time))
        logger.info(f'f1: {f1_score}')

    except Exception as e:
        logger.exception(e)
        logger.info("Completed with exception in {} seconds".format(time.time() - start_time))
        raise


if __name__ == "__main__":
    main()
