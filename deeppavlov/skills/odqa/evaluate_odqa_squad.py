"""
Evaluate ODQA on SQuAD dataset.

"""
import argparse
import time
import unicodedata
import logging
import sys
from pathlib import Path

root_path = (Path(__file__) / ".." / ".." / ".." / "..").resolve()
sys.path.append(str(root_path))

from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.metrics.squad_metrics import squad_f1

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
file = logging.FileHandler('eval_logs/odqa_squad_dev_en_ranker_top5.log')
file.setFormatter(fmt)
logger.addHandler(file)

parser = argparse.ArgumentParser()

parser.add_argument("-config_path", help="path to a JSON ranker config", type=str,
                    default='../../../deeppavlov/configs/odqa/en_odqa_infer_eval.json')
parser.add_argument("-dataset_path", help="path to a JSON formatted dataset", type=str,
                    default='/media/olga/Data/projects/ODQA/data/squad/preproc/dev-v1.1_prep_4odqa.json')


def encode_utf8(s: str):
    return unicodedata.normalize('NFD', s).encode_from_strings('utf-8')


def main():
    args = parser.parse_args()
    config = read_json(args.config_path)
    odqa = build_model_from_config(config)

    dataset = read_json(args.dataset_path)

    start_time = time.time()

    try:
        y_true_text = [instance['answers'] for instance in dataset]
        y_true_start = [instance['answers_start'] for instance in dataset]
        y_true = list(zip(y_true_text, y_true_start))

        logger.info('Making ODQA predictions...')
        questions = [instance['question'] for instance in dataset]
        y_pred = odqa(questions)

        logger.info('Counting ODQA f1 score on SQuAD...')
        f1 = squad_f1(y_true, y_pred)

        logger.info(
            'ODQA f1 score on SQuAD is: {}'.format(f1))
        logger.info("Completed successfully in {} seconds.".format(time.time() - start_time))
    except Exception as e:
        logger.exception(e)
        logger.info("Completed with exception in {} seconds".format(time.time() - start_time))
        raise


if __name__ == "__main__":
    main()
