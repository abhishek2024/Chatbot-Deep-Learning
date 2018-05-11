"""
Evaluate ranker1+ranker2+SQUAD combination on drones toloka dataset.

"""
import argparse
import time
import unicodedata
import logging
import sys
from pathlib import Path

root_path = (Path(__file__) / ".." / ".." / ".." / ".." / "..").resolve()
sys.path.append(str(root_path))

from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.metrics.squad_metrics import squad_f1

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
file = logging.FileHandler('../eval_logs/suqad_drones_toloka_ranker1-top10_ranker2-top20.log')
file.setFormatter(fmt)
logger.addHandler(file)

parser = argparse.ArgumentParser()

parser.add_argument("-config_path", help="path to a JSON ranker config", type=str,
                    default='../../../../deeppavlov/configs/squad/squad.json')
parser.add_argument("-dataset_path", help="path to a JSON formatted dataset", type=str,
                    default='/media/olga/Data/projects/ODQA/data/PMEF/qa_to_photo_toloka_revised_ranker2_output.json') # rewrite when ready


def encode_utf8(s: str):
    return unicodedata.normalize('NFD', s).encode_from_strings('utf-8')


def main():
    args = parser.parse_args()
    config = read_json(args.config_path)
    squad = build_model_from_config(config)

    dataset = read_json(args.dataset_path)

    start_time = time.time()

    try:
        y_true_text = [instance['answer'] for instance in dataset]
        y_true_start = [0 for _ in dataset]

        y_true = list(zip(y_true_text, y_true_start))
        questions_total = [instance['question'] for instance in dataset]

        contexts_total = [instance['context'][0] for instance in dataset]

        BATCH_SIZE = 10

        question_batches = [questions_total[x:x + BATCH_SIZE] for x in
                            range(0, len(questions_total), BATCH_SIZE)]
        context_batches = [contexts_total[x:x + BATCH_SIZE] for x in
                           range(0, len(contexts_total), BATCH_SIZE)]
        len_batches = len(question_batches)

        y_pred = []

        for i, zipped in enumerate(zip(question_batches, context_batches)):
            logger.info('Making SQUAD predictions on batch {} of {}'.format(i, len_batches))
            questions = zipped[0]
            contexts = zipped[1]
            squad_input = list(zip(contexts, questions))
            y_pred_batch = squad(squad_input)
            y_pred += y_pred_batch

        logger.info('Counting SQUAD f1 score on toloka dataset...')
        f1 = squad_f1(y_true, y_pred)
        logger.info('ODQA total f1 score on SQuAD is {}'.format(f1))
        logger.info("Completed successfully in {} seconds.".format(time.time() - start_time))
    except Exception as e:
        logger.exception(e)
        logger.info("Completed with exception in {} seconds".format(time.time() - start_time))
        raise


if __name__ == "__main__":
    main()