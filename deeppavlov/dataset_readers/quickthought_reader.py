import random
from pathlib import Path

from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.core.common.registry import register
from deeppavlov.core.commands.utils import expand_path as expand_path_fn
from deeppavlov.core.commands.train import train_evaluate_model_from_config


@register('quickthought_reader')
class QuickThoughtReader(DatasetReader):
    """
    Data description:
    
    Directory with train_fname and valid_fname text files
    each file consist of \n-separated sentences
    sentences = tokens separated by spaces
    different documents are separated by blank line (\n)

    Example:
    Taapaca is a Holocene volcanic complex in northern Chile 's Arica y Parinacota Region .
    Located in the Chilean segment of the Andean volcanic chain , it is part of ...

    BepiColombo is a joint mission of the European Space Agency ( ESA ) and the Japan Aerospace ...

    Output:
        dict with keys 'train' and 'valid',
        'train': list((sentence, next_sentence), unique_id)
        'valid': list((sentence, possibly_next_sentence), is_really_next)
    """
    def read(self, data_path,
             train_fname='train.txt', valid_fname='valid.txt', expand_path=True, seed=42,
             *args, **kwargs):
        self.seed = seed

        data_path = Path(data_path)
        if expand_path:
            data_path = expand_path_fn(data_path)
        dataset = {'train': None, 'valid': None}

        train_path = data_path / train_fname
        valid_path = data_path / valid_fname

        dataset['train'] = self._make_sentences_pairs_train(train_path)
        dataset['valid'] = self._make_sentences_pairs_valid(valid_path)

        return dataset

    def _make_sentences_pairs_train(self, path):
        sentences_pairs = []
        last_sentence = None
        with open(path, 'r', encoding="utf-8") as f:
            for i, current_sentence in enumerate(f):
                current_sentence = current_sentence.rstrip('\n')
                if not current_sentence:  # document delimiter
                    last_sentence = None
                    continue

                if last_sentence is None:
                    last_sentence = current_sentence
                else:
                    # i for triplet loss - different i for different pairs
                    current_pair = (last_sentence, current_sentence), i
                    last_sentence = current_sentence
                    sentences_pairs.append(current_pair)
        return sentences_pairs

    def _make_sentences_pairs_valid(self, path, frac_false=0.5):
        """
        Valid dataset should have labeling with whever or not
        second sentence in pair is truly next sentence in the text
        """
        random.seed(self.seed)

        sentences_pairs = []
        all_sentences = []
        last_sentence = None
        with open(path, 'r', encoding="utf-8") as f:
            for i, current_sentence in enumerate(f):
                current_sentence = current_sentence.rstrip('\n')
                if not current_sentence:  # document delimiter
                    last_sentence = None
                    continue
                all_sentences.append(current_sentence)

                if last_sentence is None:
                    last_sentence = current_sentence
                else:
                    # all pairs are true --> 1
                    current_pair = (last_sentence, current_sentence), 1
                    last_sentence = current_sentence
                    sentences_pairs.append(current_pair)

        sentences_pairs_labeled = []
        # make negative samples
        for (sent1, sent2), _ in sentences_pairs:
            if random.random() < frac_false:
                pair = (sent1, random.choice(all_sentences)), 0
            else:
                pair = (sent1, sent2), 1
            sentences_pairs_labeled.append(pair)

        return sentences_pairs_labeled
