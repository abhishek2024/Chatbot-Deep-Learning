import logging
import argparse
from operator import itemgetter
import sys
from pathlib import Path

root_path = (Path(__file__) / ".." / ".." / ".." / ".." / ".." / "..").resolve()
sys.path.append(str(root_path))

from deeppavlov.core.common.file import read_json

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
file = logging.FileHandler('neg_sampling_analyse.log')
file.setFormatter(fmt)
logger.addHandler(file)

parser = argparse.ArgumentParser()

parser.add_argument("-dataset_path", help="path to a SQuAD dataset preprocessed for ranker", type=str,
                    default="/media/olga/Data/datasets/squad/preproc/dev-v1.1_negative_samples.json")


def answer_found(answers, context):
    for a in answers:
        if a in context:
            return True
    return False


def hard_neg_sample_found(contexts):
    score_idx = 2
    scores = list(map(itemgetter(score_idx), contexts))
    return any(s <= 0.0001 for s in scores)


def main():
    args = parser.parse_args()
    dataset = read_json(args.dataset_path)
    # dataset = dataset[:3]
    logger.info(f"Dataset size: {len(dataset)}")

    count_answers_found = 0
    count_neg_samples = 0
    unique_contexts = set()
    unique_neg_samples = set()

    for i, instance in enumerate(dataset):
        question = instance['question']
        answers = instance['answers']
        contexts = instance['contexts']
        ground_context = contexts[0]
        other_contexts = contexts[1:]

        if answer_found(answers, other_contexts[0]):
            logger.info("Found answer for the following data:")
            logger.info(question)
            logger.info(answers)
            logger.info(ground_context)
            logger.info(f"Instance index: {i}")
            count_answers_found += 1

        if hard_neg_sample_found(other_contexts):
            count_neg_samples += 1

        unique_contexts = unique_contexts.union(set(map(itemgetter(1), other_contexts)))
        unique_contexts.add(ground_context)

        unique_neg_samples = unique_neg_samples.union(set(list(c[1] for c in other_contexts if c[2] <= 0.0001)))

        logger.info(f"Processed instance {i}")

        # logger.info(f"Hard negative samples found: {count_neg_samples}")
        # logger.info(f"Answers found in {count_answers_found} instances.")

    logger.info(f"Total hard negative samples found: {count_neg_samples}")
    logger.info(f"Total answers found in {count_answers_found} instances.")
    logger.info(f"Total number of unique contexts: {len(unique_contexts)}.")
    logger.info(f"Total number of unique hard negative samples: {len(unique_neg_samples)}.")


if __name__ == "__main__":
    main()
