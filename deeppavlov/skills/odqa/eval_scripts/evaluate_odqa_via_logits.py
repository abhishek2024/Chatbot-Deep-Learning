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
from itertools import chain
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
file = logging.FileHandler('../eval_logs/odqa_f1_logits.log')
file.setFormatter(fmt)
logger.addHandler(file)

parser = argparse.ArgumentParser()

parser.add_argument("-ranker_config_path", help="path to a JSON ranker config", type=str,
                    default='../../../../deeppavlov/configs/odqa/en_ranker_tfidf_wiki.json')
parser.add_argument("-dataset_path", help="path to SQuAD dataset", type=str,
                    default='/home/gureenkova/data/squad/dev-v1.1_prep_4odqa.json')
parser.add_argument("-reader_config_path", help="path to squad reader config", type=str,
                    default='../../../../deeppavlov/configs/squad/squad.json')


def main():
    args = parser.parse_args()
    ranker_config = read_json(args.ranker_config_path)
    reader_config = read_json(args.reader_config_path)
    ranker = build_model_from_config(ranker_config)  # chainer
    reader = build_model_from_config(reader_config)
    # print(reader([("Deep Pavlov killed Kenny.", "Who killed Kenny?")]))
    dataset = read_json(args.dataset_path)
    # dataset = dataset[:3]

    qa_dataset_size = len(dataset)
    logger.info('QA dataset size: {}'.format(qa_dataset_size))
    # n_queries = 0  # DEBUG
    start_time = time.time()
    TEXT_IDX = 1
    SQUAD_LIMIT = 500
    SQUAD_BATCH_SIZE = 5

    try:
        mapping = {}

        y_true_text = [instance['answers'] for instance in dataset]
        y_true_start = [instance['answers_start'] for instance in dataset]
        y_true = list(zip(y_true_text, y_true_start))

        ranker_answers = ranker([i['question'] for i in dataset])
        returned_db_size = len(ranker_answers[0])
        logger.info("Returned DB size: {}".format(returned_db_size))

        for n in [5, 10]:
            # print(f"Starting for {n}")
            y_pred_on_i = []
            i = 1
            for qa, ranker_answer in zip(dataset, ranker_answers):
                question = qa['question']
                texts = [answer[TEXT_IDX] for answer in ranker_answer[:n]]

                # make paragraphs
                paragraphs = []
                for t in texts:
                    sentences = sent_tokenize(t)
                    chunks = make_chunks(sentences, SQUAD_LIMIT)
                    paragraphs += chunks

                questions = [question] * len(paragraphs)
                
                reader_feed = list(zip(paragraphs, questions))
                batches = [reader_feed[x:x + SQUAD_BATCH_SIZE] for x in range(0, len(reader_feed), SQUAD_BATCH_SIZE)]
                y_pred = []
                for batch in batches:
                     y_pred += reader(batch)
                 
                best_y_pred = sorted(y_pred, key=itemgetter(3), reverse=True)[0][:2]
                # print(best_y_pred)
                y_pred_on_i.append(best_y_pred)
                logger.info(
                    "Processed {} qa pairs for n={}".format(i, n))
                i += 1
            # print(y_true)
            # print(y_pred_on_i)
            total_f1_on_top_i = squad_f1(y_true, y_pred_on_i)
            logger.info(
                'ODQA f1 for top {}: {}'.format(n, total_f1_on_top_i))
            mapping[n] = total_f1_on_top_i

        logger.info("Completed successfully in {} seconds.".format(time.time() - start_time))
        logger.info("Quality mapping: {}".format(mapping))

    except Exception as e:
        logger.exception(e)
        logger.info("Completed with exception in {} seconds".format(time.time() - start_time))
        raise


if __name__ == "__main__":
    main()
