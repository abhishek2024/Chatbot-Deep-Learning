#  transformer_chatbot
#  Copyright (C) 2018 Golovanov, Tselousov
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import spacy
import ftfy

from pytorch_pretrained_bert import BertTokenizer


class SpacyLowerTokenizer:
    def __init__(self):
        self.tokenizer = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])

    def __call__(self, string):
        string = ftfy.fix_text(string)
        words = [t.text.strip() for t in self.tokenizer(string)]
        words = [w.lower() for w in words if w]

        return words


class BPEVocab:
    we = '</w>'

    pad_token = '<pad>'
    bos_token = '<s>'
    eos_token = '</s>'
    info_bos = '<i>'
    info_eos = '</i>'
    talker1_bos = '<t1>'
    talker1_eos = '</t1>'
    talker2_bos = '<t2>'
    talker2_eos = '</t2>'

    @staticmethod
    def from_files(vocab_path, codes_path, *args, **kwargs):
        with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
            vocab = [t.strip() for t in vocab_file.readlines()]

        with open(codes_path, 'r', encoding='utf-8') as codes_file:
            codes = [c.strip() for c in codes_file.readlines()]

            if codes[0].startswith('#version'):
                codes = codes[1:]

            codes = [tuple(c.split()) for c in codes if c]

        return BPEVocab(vocab, codes, *args, **kwargs)

    @staticmethod
    def get_pairs(string):
        if len(string) < 2:
            return set()

        return set(zip(string[:-1], string[1:]))

    def __init__(self, vocab, codes, tokenizer=SpacyLowerTokenizer()):
        # TODO: add check for special tokens
        self.spec_tokens = [BPEVocab.pad_token, BPEVocab.bos_token, BPEVocab.eos_token,
                            BPEVocab.info_bos, BPEVocab.info_eos, BPEVocab.talker1_bos,
                            BPEVocab.talker1_eos, BPEVocab.talker2_bos, BPEVocab.talker2_eos]
        vocab = self.spec_tokens + vocab

        self.token2id = {t: i for i, t in enumerate(vocab)}
        self.id2token = {i: t for i, t in enumerate(vocab)}
        self.bpe_ranks = dict(zip(codes, range(len(codes))))
        self.tokenizer = tokenizer
        self.cache = {}

    def __len__(self):
        return len(self.token2id)

    @property
    def n_special_tokens(self):
        return len(self.spec_tokens)

    @property
    def special_tokens_ids(self):
        return [self.token2id[t] for t in self.spec_tokens]

    @property
    def pad_id(self):
        return self.token2id[BPEVocab.pad_token]

    @property
    def bos_id(self):
        return self.token2id[BPEVocab.bos_token]

    @property
    def eos_id(self):
        return self.token2id[BPEVocab.eos_token]

    @property
    def info_bos_id(self):
        return self.token2id[BPEVocab.info_bos]

    @property
    def info_eos_id(self):
        return self.token2id[BPEVocab.info_eos]

    @property
    def talker1_bos_id(self):
        return self.token2id[BPEVocab.talker1_bos]

    @property
    def talker1_eos_id(self):
        return self.token2id[BPEVocab.talker1_eos]

    @property
    def talker2_bos_id(self):
        return self.token2id[BPEVocab.talker2_bos]

    @property
    def talker2_eos_id(self):
        return self.token2id[BPEVocab.talker2_eos]

    @property
    def sep_id(self):
        return False

    def _bpe(self, token):
        if token in self.cache:
            return self.cache[token]

        word = tuple(token[:-1]) + (token[-1] + BPEVocab.we,)
        pairs = BPEVocab.get_pairs(word)

        if not pairs:
            return (token + BPEVocab.we,)

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)

            if len(word) == 1:
                break
            else:
                pairs = BPEVocab.get_pairs(word)

        self.cache[token] = word

        return word

    def string2ids(self, string):
        tokens = self.tokenizer(string)
        bpe_tokens = sum([self._bpe(t) for t in tokens], tuple())
        ids = [self.token2id[t] for t in bpe_tokens if t in self.token2id]

        return ids

    def ids2string(self, ids):
        bpe_tokens = [self.id2token[id] for id in ids]

        return ''.join(bpe_tokens).replace(BPEVocab.we, ' ')


class BertBPEVocab:
    we = '##'

    pad_token = '[PAD]'
    bos_token = '[S]'
    eos_token = '[/S]'
    info_bos = '[I]'
    info_eos = '[/I]'
    talker1_bos = '[T1]'
    talker1_eos = '[/T1]'
    talker2_bos = '[T2]'
    talker2_eos = '[/T2]'
    talker2_eos = '[/T2]'
    unk_token = '[UNK]'
    cls_token = '[CLS]'
    sep_token = '[SEP]'
    mask_token = '[MASK]'

    @staticmethod
    def from_files(vocab_path, *args, **kwargs):
        tokenizer = BertTokenizer(vocab_path, do_lower_case=False)
        return BertBPEVocab(tokenizer, *args, **kwargs)

    @staticmethod
    def get_pairs(string):
        if len(string) < 2:
            return set()

        return set(zip(string[:-1], string[1:]))

    def __init__(self, tokenizer):
        # TODO: add check for special tokens
        tokenizer
        self.spec_tokens = [BertBPEVocab.pad_token, BertBPEVocab.bos_token, BertBPEVocab.eos_token,
                            BertBPEVocab.info_bos, BertBPEVocab.info_eos, BertBPEVocab.talker1_bos,
                            BertBPEVocab.talker1_eos, BertBPEVocab.talker2_bos, BertBPEVocab.talker2_eos,
                            BertBPEVocab.unk_token, BertBPEVocab.cls_token, BertBPEVocab.sep_token,
                            BertBPEVocab.mask_token,
                            ]
        self.token2id = tokenizer.vocab
        self.id2token = tokenizer.ids_to_tokens
        self.tokenizer = tokenizer
        self.cache = {}

    def __len__(self):
        return len(self.token2id)

    @property
    def n_special_tokens(self):
        return len(self.spec_tokens)

    @property
    def special_tokens_ids(self):
        return [self.token2id[t] for t in self.spec_tokens]

    @property
    def pad_id(self):
        return self.token2id[BertBPEVocab.pad_token]

    @property
    def bos_id(self):
        return self.token2id[BertBPEVocab.bos_token]

    @property
    def eos_id(self):
        return self.token2id[BertBPEVocab.eos_token]

    @property
    def info_bos_id(self):
        return self.token2id[BertBPEVocab.info_bos]

    @property
    def info_eos_id(self):
        return self.token2id[BertBPEVocab.info_eos]

    @property
    def talker1_bos_id(self):
        return self.token2id[BertBPEVocab.talker1_bos]

    @property
    def talker1_eos_id(self):
        return self.token2id[BertBPEVocab.talker1_eos]

    @property
    def talker2_bos_id(self):
        return self.token2id[BertBPEVocab.talker2_bos]

    @property
    def talker2_eos_id(self):
        return self.token2id[BertBPEVocab.talker2_eos]

    @property
    def sep_id(self):
        return self.token2id[BertBPEVocab.sep_token]

    def string2ids(self, string):
        tokens = self.tokenizer.tokenize(string)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # tokens = self.tokenizer.convert_ids_to_tokens(tokens)
        return ids

    def ids2string(self, ids):
        bpe_tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return ' '.join(bpe_tokens).replace(' '+BertBPEVocab.we, '')
