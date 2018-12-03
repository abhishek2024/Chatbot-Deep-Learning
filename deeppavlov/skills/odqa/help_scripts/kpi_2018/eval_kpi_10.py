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
import random
import csv

root_path = (Path(__file__) / ".." / ".." / ".." / ".." / "..").resolve()
sys.path.append(str(root_path))
sys.path.append('/media/olga/Data/projects/DeepPavlov')


from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.core.common.file import read_json
from deeppavlov.metrics.squad_metrics import squad_f1, exact_match

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt=fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()

parser.add_argument("-dataset_path", help="path to KPI_4 dataset", type=str,
                    default='/media/olga/Data/datasets/kpi_2018/kpi10_train-v1.1_test.json')
parser.add_argument("-config_path", help="path to ODQA config", type=str,
                    default='/media/olga/Data/projects/DeepPavlov/deeppavlov/configs/odqa/en_odqa_tfidf_wiki_conversation_mode_pop.json')
parser.add_argument("-output_path", help="path to ODQA config", type=str,
                    default='kpi_10_sample_50_test.csv')


def main():
    args = parser.parse_args()
    odqa = build_model_from_config(read_json(args.config_path))
    dataset = read_json(args.dataset_path)
    dataset = dataset['data'][0]['paragraphs']
    # dataset = dataset[:3]

    # Select 20 random samples:
    # dataset = random.sample(dataset, 50)

    qa_dataset_size = len(dataset)
    logger.info('QA dataset size: {}'.format(qa_dataset_size))
    start_time = time.time()

    try:

        y_true_text = []
        y_true_start = []
        questions = []

        with open(args.output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Question', 'True Answer', 'Predicted Answer'])  # add header

            for instance in dataset:
                qas = instance['qas']
                for j in range(len(qas)):
                    answers = qas[j]['answers']
                    question = qas[j]['question']
                    for k in range(len(answers)):
                        y_true_text.append(answers[k]['text'])
                        y_true_start.append(answers[k]['answer_start'])
                        questions.append(question)

            # y_true = list(zip(y_true_text, y_true_start))
            y_true_text = [[el] if not isinstance(el, list) else el for el in y_true_text]
            y_true_start = [[el] if not isinstance(el, list) else el for el in y_true_start]
            y_true = list(zip(y_true_text, y_true_start))

            y_pred = []
            for i, q in enumerate(questions):
                logger.info(f'Processing question {i}')
                answer = odqa([q])[0][0]
                y_pred.append((answer[0], answer[1]))
                logger.info(f"Current KPI 10 f1: {squad_f1(y_true[:i+1], y_pred)}")
                logger.info(f"Current KPI 10 em: {exact_match(y_true[:i+1], y_pred)}")
                writer.writerow([q, y_true[i][0][0], answer[0]])

            f1 = squad_f1(y_true, y_pred)
            em = exact_match(y_true, y_pred)
            logger.info(f"Total KPI 10 f1: {f1}")
            logger.info(f"Total KPI 10 em: {em}")
            logger.info("Completed successfully in {} seconds.".format(time.time() - start_time))

    except Exception as e:
        logger.exception(e)
        logger.info("Completed with exception in {} seconds".format(time.time() - start_time))
        raise


if __name__ == "__main__":
    main()
