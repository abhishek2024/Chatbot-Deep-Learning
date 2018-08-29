import logging
import argparse
import json
import sys
from pathlib import Path

root_path = (Path(__file__) / ".." / ".." / ".." / ".." / ".." / "..").resolve()
sys.path.append(str(root_path))

from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.infer import build_model_from_config

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
file = logging.FileHandler('neg_sampling.log')
file.setFormatter(fmt)
logger.addHandler(file)

parser = argparse.ArgumentParser()

parser.add_argument("-new_dataset_path", help="path to a SQuAD dataset preprocessed for ranker", type=str,
                    default="/media/olga/Data/datasets/squad/preproc/train-v1.1_negative_samples.json")
parser.add_argument("-dataset_path", help="path to a SQuAD dataset preprocessed for ranker", type=str,
                    default="/media/olga/Data/datasets/squad/preproc/train-v1.1_prep_4ranker.json")
parser.add_argument("-ranker_config_path", help="path to a tfidf ranker config", type=str,
                    default="/media/olga/Data/projects/DeepPavlov/deeppavlov/configs"
                            "/odqa/en_ranker_tfidf_train_chunk_db.json")


def answer_found(question, answers, context):
    for a in answers:
        if a in context:
            logger.info("Found answer for the following data:")
            logger.info(question)
            logger.info(answers)
            logger.info(context)
            return True
    return False


def main():
    args = parser.parse_args()
    dataset = read_json(args.dataset_path)
    # dataset = dataset[:3]
    print(f"Dataset size: {len(dataset)}")

    ranker = build_model_from_config(read_json(args.ranker_config_path))

    new_dataset = []

    for i, instance in enumerate(dataset):
        question = instance['question']
        answers = instance['answers']
        context = instance['context']

        retrieved_contexts = ranker([question])[0]
        # retrieved_contexts = [rc for rc in retrieved_contexts if not answer_found(question, answers, rc)]

        new_dataset.append({"question": question,
                            "answers": answers,
                            "contexts": [context, *retrieved_contexts]})

        logger.info(f"Processed instance {i}")

    with open(args.new_dataset_path, 'w') as fin:
        json.dump(new_dataset, fin)


if __name__ == "__main__":
    main()
