from pathlib import Path
import csv
import re
import random
from collections import defaultdict

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader


@register('negative_sber_faq_reader')
class NegativeSberFAQReader(DatasetReader):

    def read(self, data_path):
        dataset = {'train': None, 'valid': None, 'test': None}
        train_fname = Path(data_path) / 'sber_faq_train_50.csv'
        valid_fname = Path(data_path) / 'sber_faq_val_50.csv'
        test_fname = Path(data_path) / 'sber_faq_test_50.csv'
        self.sen2int_vocab = {}
        self.classes_vocab_train = {}
        self.classes_vocab_valid = {}
        self.classes_vocab_test = {}
        dataset["train"] = self.preprocess_data(train_fname)
        dataset["valid"] = self.preprocess_data(valid_fname, type='valid', num_ranking_samples=9)
        dataset["test"] = self.preprocess_data(test_fname, type='test', num_ranking_samples=9)
        return dataset

    def preprocess_data(self, fname, type='train', num_ranking_samples=1):
        data = []
        classes_vocab = defaultdict(set)
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for el in reader:
                classes_vocab[int(el[1])].add(self.clean_sen(el[0]))
        for k, v in classes_vocab.items():
            sen_li = list(v)
            pos_pairs = self._get_pairs(sen_li)
            neg_resps = self._get_neg_resps(classes_vocab, k, num_neg_resps=num_ranking_samples * len(pos_pairs))
            neg_resps = [neg_resps[i:i + num_ranking_samples] for i in range(0, len(neg_resps), num_ranking_samples)]
            if type == 'train':
                data += list(zip(pos_pairs, list(k*np.ones(len(pos_pairs), dtype='int'))))
            else:
                lists = [el[0] + el[1] for el in zip(pos_pairs, neg_resps)]
                data += list(zip(lists, list(np.ones(len(lists), dtype='int'))))
        return data

    def clean_sen(self, sen):
        return re.sub('\[Клиент:.*\]', '', sen, flags=re.IGNORECASE). \
            replace('&amp, laquo, ', '').replace('&amp, raquo, ', ''). \
            replace('&amp laquo ', '').replace('&amp raquo ', ''). \
            replace('&amp quot ', '').replace('&amp, quot, ', '').strip()

    def _get_neg_resps(self, classes_vocab, label, num_neg_resps):
        neg_resps = []
        for k, v in classes_vocab.items():
            if k != label:
                neg_resps.append(list(v))
        neg_resps = [x for el in neg_resps for x in el]
        neg_resps = random.choices(neg_resps, k=num_neg_resps)
        return neg_resps

    def _get_pairs(self, sen_list):
        pairs = []
        for i in range(len(sen_list) - 1):
            for j in range(i + 1, len(sen_list)):
                pairs.append([sen_list[i], sen_list[j]])
                pairs.append([sen_list[j], sen_list[i]])
        return pairs