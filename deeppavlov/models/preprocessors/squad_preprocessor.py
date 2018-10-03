"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pickle
import unicodedata
from collections import Counter
from pathlib import Path
import string

import numpy as np
from nltk import word_tokenize
from tqdm import tqdm
import lightgbm as lgbm
import pandas as pd
import json

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import download
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.estimator import Estimator

logger = get_logger(__name__)


@register('squad_preprocessor')
class SquadPreprocessor(Component):
    def __init__(self, context_limit, question_limit, char_limit, *args, **kwargs):
        self.context_limit = context_limit
        self.question_limit = question_limit
        self.char_limit = char_limit

    def __call__(self, contexts_raw, questions_raw, **kwargs):
        contexts = []
        contexts_tokens = []
        contexts_chars = []
        contexts_r2p = []
        contexts_p2r = []
        questions = []
        questions_tokens = []
        questions_chars = []
        spans = []
        for c_raw, q_raw in zip(contexts_raw, questions_raw):
            c, r2p, p2r = SquadPreprocessor.preprocess_str(c_raw, return_mapping=True)
            c_tokens = [token.replace("''", '"').replace("``", '"') for token in word_tokenize(c)][:self.context_limit]
            c_chars = [list(token)[:self.char_limit] for token in c_tokens]
            q = SquadPreprocessor.preprocess_str(q_raw)
            q_tokens = [token.replace("''", '"').replace("``", '"') for token in word_tokenize(q)][:self.question_limit]
            q_chars = [list(token)[:self.char_limit] for token in q_tokens]
            contexts.append(c)
            contexts_tokens.append(c_tokens)
            contexts_chars.append(c_chars)
            contexts_r2p.append(r2p)
            contexts_p2r.append(p2r)
            questions.append(q)
            questions_tokens.append(q_tokens)
            questions_chars.append(q_chars)
            spans.append(SquadPreprocessor.convert_idx(c, c_tokens))
        return contexts, contexts_tokens, contexts_chars, contexts_r2p, contexts_p2r, \
               questions, questions_tokens, questions_chars, spans

    @staticmethod
    def preprocess_str(line, return_mapping=False):
        """ Removes unicode and other characters from str

        Args:
            line:
            return_mapping: return mapping from line to preprocessed line or not

        Returns:
            preprocessed line, raw2preprocessed, preprocessed2raw

        """
        line = line.replace("''", '" ').replace("``", '" ')
        if not return_mapping:
            return ''.join(c for c in line if not unicodedata.combining(c))

        r2p = [len(line)] * (len(line) + 1)
        p2r = [len(line)] * (len(line) + 1)
        s = ''
        for i, c in enumerate(line):
            if unicodedata.combining(c):
                r2p[i] = -1
            else:
                s += c
                r2p[i] = len(s) - 1
                p2r[len(s) - 1] = i
        return s, r2p, p2r

    @staticmethod
    def convert_idx(text, tokens):
        current = 0
        spans = []
        for token in tokens:
            current = text.find(token, current)
            if current < 0:
                print("Token {} cannot be found".format(token))
                raise Exception()
            spans.append((current, current + len(token)))
            current += len(token)
        return spans


@register('squad_ans_preprocessor')
class SquadAnsPreprocessor(Component):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, answers_raw, answers_start, r2ps, spans, **kwargs):
        """ Processes answers for SQuAD dataset

        Args:
            answers_raw: list of str [batch_size x number_of_answers]
            answers_start: start position of answer (in chars) [batch_size x number_of_answers]
            r2ps: mapping from raw context to processed context
            spans: mapping for tokens in context to position in context
            **kwargs:

        Returns:
            processed answer text, start position in tokens, end position in tokens
            [batch_size x number_of_answers]

        """
        answers = []
        start = []
        end = []
        for ans_raw, ans_st, r2p, span in zip(answers_raw, answers_start, r2ps, spans):
            start.append([])
            end.append([])
            answers.append([])
            for a_raw, a_st in zip(ans_raw, ans_st):
                ans = SquadPreprocessor.preprocess_str(a_raw)
                ans_st = r2p[a_st]
                ans_end = ans_st + len(ans)
                answer_span = []
                for idx, sp in enumerate(span):
                    if not (ans_end <= sp[0] or ans_st >= sp[1]):
                        answer_span.append(idx)
                if len(answer_span) != 0:
                    y1, y2 = answer_span[0], answer_span[-1]
                else:
                    # answer not found in context
                    y1, y2 = -1, -1
                start[-1].append(y1)
                end[-1].append(y2)
                answers[-1].append(ans)
        return answers, start, end


@register('squad_vocab_embedder')
class SquadVocabEmbedder(Estimator):
    def __init__(self, emb_folder, emb_url, save_path, load_path,
                 context_limit, question_limit, char_limit, level='token', *args, **kwargs):
        self.emb_folder = expand_path(emb_folder)
        self.level = level
        self.emb_url = emb_url
        self.emb_file_name = Path(emb_url).name
        self.save_path = expand_path(save_path)
        self.load_path = expand_path(load_path)
        self.context_limit = context_limit
        self.question_limit = question_limit
        self.char_limit = char_limit
        self.loaded = False

        self.NULL = "<NULL>"
        self.OOV = "<OOV>"

        self.emb_folder.mkdir(parents=True, exist_ok=True)

        if not (self.emb_folder / self.emb_file_name).exists():
            download(self.emb_folder / self.emb_file_name, self.emb_url)

        if self.load_path.exists():
            self.load()

    def __call__(self, contexts, questions):
        if self.level == 'token':
            c_idxs = np.zeros([len(contexts), self.context_limit], dtype=np.int32)
            q_idxs = np.zeros([len(questions), self.question_limit], dtype=np.int32)
            for i, context in enumerate(contexts):
                for j, token in enumerate(context[:self.context_limit]):
                    c_idxs[i, j] = self._get_idx(token)

            for i, question in enumerate(questions):
                for j, token in enumerate(question[:self.question_limit]):
                    q_idxs[i, j] = self._get_idx(token)

        elif self.level == 'char':
            c_idxs = np.zeros([len(contexts), self.context_limit, self.char_limit], dtype=np.int32)
            q_idxs = np.zeros([len(questions), self.question_limit, self.char_limit], dtype=np.int32)
            for i, context in enumerate(contexts):
                for j, token in enumerate(context[:self.context_limit]):
                    for k, char in enumerate(token[:self.char_limit]):
                        c_idxs[i, j, k] = self._get_idx(char)

            for i, question in enumerate(questions):
                for j, token in enumerate(question[:self.question_limit]):
                    for k, char in enumerate(token[:self.char_limit]):
                        q_idxs[i, j, k] = self._get_idx(char)

        return c_idxs, q_idxs

    def fit(self, contexts, questions, *args, **kwargs):
        self.vocab = Counter()
        self.embedding_dict = dict()
        if not self.loaded:
            logger.info('SquadVocabEmbedder: fitting with {}s'.format(self.level))
            if self.level == 'token':
                for line in tqdm(contexts + questions):
                    for token in line:
                        self.vocab[token] += 1
            elif self.level == 'char':
                for line in tqdm(contexts + questions):
                    for token in line:
                        for c in token:
                            self.vocab[c] += 1
            else:
                raise RuntimeError("SquadVocabEmbedder::fit: Unknown level: {}".format(self.level))

            with (self.emb_folder / self.emb_file_name).open('r', encoding='utf8') as femb:
                emb_voc_size, self.emb_dim = map(int, femb.readline().split())
                for line in tqdm(femb, total=emb_voc_size):
                    line_split = line.strip().split(' ')
                    word = line_split[0]
                    vec = np.array(line_split[1:], dtype=float)
                    if len(vec) != self.emb_dim:
                        continue
                    if word in self.vocab:
                        self.embedding_dict[word] = vec

            self.token2idx_dict = {token: idx for idx,
                                             token in enumerate(self.embedding_dict.keys(), 2)}
            self.token2idx_dict[self.NULL] = 0
            self.token2idx_dict[self.OOV] = 1
            self.embedding_dict[self.NULL] = [0. for _ in range(self.emb_dim)]
            self.embedding_dict[self.OOV] = [0. for _ in range(self.emb_dim)]
            idx2emb_dict = {idx: self.embedding_dict[token]
                            for token, idx in self.token2idx_dict.items()}
            self.emb_mat = np.array([idx2emb_dict[idx] for idx in range(len(idx2emb_dict))])

    def load(self, *args, **kwargs):
        logger.info('SquadVocabEmbedder: loading saved {}s vocab from {}'.format(self.level, self.load_path))
        self.emb_dim, self.emb_mat, self.token2idx_dict = pickle.load(self.load_path.open('rb'))
        self.loaded = True

    def save(self, *args, **kwargs):
        logger.info('SquadVocabEmbedder: saving {}s vocab to {}'.format(self.level, self.save_path))
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump((self.emb_dim, self.emb_mat, self.token2idx_dict), self.save_path.open('wb'))

    def _get_idx(self, el):
        """ Returns idx for el (token or char).

        Args:
            el: token or character

        Returns:
            idx in vocabulary
        """
        for e in (el, el.lower(), el.capitalize(), el.upper()):
            if e in self.token2idx_dict:
                return self.token2idx_dict[e]
        return 1


@register('squad_ans_postprocessor')
class SquadAnsPostprocessor(Component):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, ans_start, ans_end, contexts, p2rs, spans, **kwargs):
        """ Postprocesses predicted answers for SQuAD dataset

        Args:
            ans_start: predicted start position in processed context: list of ints with len(ans_start) == batch_size
            ans_end: predicted end position in processed context
            contexts: raw contexts
            p2rs: mapping from processed context to raw
            spans: tokens positions in context
            **kwargs:

        Returns:
            postprocessed answer text, start position in raw context, end position in raw context
        """
        answers = []
        starts = []
        ends = []
        for a_st, a_end, c, p2r, span in zip(ans_start, ans_end, contexts, p2rs, spans):
            if isinstance(a_st, int) or isinstance(a_st, np.int64) or isinstance(a_st, np.int32):
                start, end, answer = self._get_answer(a_st, a_end, p2r, span, c)
                answers.append(answer)
                starts.append(start)
                ends.append(end)
            else:  # we got list of top_k answers
                top_answers = []
                top_starts = []
                top_ends = []
                for a_st, a_end in zip(a_st, a_end):
                    start, end, answer = self._get_answer(a_st, a_end, p2r, span, c)
                    top_answers.append(answer)
                    top_starts.append(start)
                    top_ends.append(end)
                answers.append(top_answers)
                starts.append(top_starts)
                ends.append(top_ends)
        return answers, starts, ends

    @staticmethod
    def _get_answer(a_st, a_end, p2r, span, context):
        if a_st == -1 or a_end == -1:
            start = -1
            end = -1
            answer = ''
        else:
            start = p2r[span[a_st][0]]
            end = p2r[span[a_end][1]]
            answer = context[start:end]
        return start, end, answer


@register('squad_lgbm_scorer')
class SquadLGBMScorer(Estimator):
    def __init__(self, save_path, load_path, *args, **kwargs):
        self.save_path = expand_path(save_path)
        self.load_path = expand_path(load_path)
        self.model = None

        if self.load_path.exists():
            self.load()
        else:
            raise RuntimeError('SquadLGBMScorer: there is no pretrained model')

    def fit(self, *args, **kwargs):
        pass

    def __call__(self, prob, score):
        df = pd.DataFrame()
        data = list(zip(prob, score))
        df['prob'] = [p for p, s in data]
        df['score'] = [s for p, s in data]
        df['log prob'] = [np.log(p) for p, s in data]
        df['log score'] = [np.log(s) for p, s in data]
        scores = self.model.predict(df)
        return scores

    def load(self, *args, **kwargs):
        logger.info('SquadLGBMScorer: loading saved model vocab from {}'.format(self.load_path))
        self.model = lgbm.Booster(model_file=str(self.load_path))

    def save(self, *args, **kwargs):
        pass


@register('squad_features_extractor')
class SquadFeaturesExtractor(Component):
    """
    Extracts exact match and token frequency features from context and question.
    """
    def __init__(self, context_limit, question_limit, *args, **kwargs):
        self.context_limit = context_limit
        self.question_limit = question_limit
        self.punkt = set(string.punctuation)

    def __call__(self, contexts_tokens, questions_tokens, **kwargs):
        c_em = np.zeros([len(contexts_tokens), self.context_limit], dtype=np.float32)
        c_tf = np.zeros([len(contexts_tokens), self.context_limit], dtype=np.float32)
        q_em = np.zeros([len(questions_tokens), self.question_limit], dtype=np.float32)
        q_tf = np.zeros([len(questions_tokens), self.question_limit], dtype=np.float32)
        for i, (context_tokens, question_tokens) in enumerate(zip(contexts_tokens, questions_tokens)):
            context_tokens = context_tokens[:self.context_limit]
            question_tokens = question_tokens[:self.question_limit]
            context_tokens = list(map(lambda x: x.lower(), context_tokens))
            question_tokens = list(map(lambda x: x.lower(), question_tokens))
            context_len = len(context_tokens)
            question_len = len(question_tokens)

            if context_len == 0 or question_len == 0:
                raise RuntimeError('Context or question is empty.')

            context_unique_tokens = Counter(context_tokens)
            question_unique_tokens = Counter(question_tokens)
            for j, c_token in enumerate(context_tokens):
                c_em[i][j] = 1 if c_token in question_unique_tokens and c_token not in self.punkt else 0
                c_tf[i][j] = context_unique_tokens[c_token] / context_len if c_token not in self.punkt else 0
            for j, q_token in enumerate(question_tokens):
                q_em[i][j] = 1 if q_token in context_unique_tokens and q_token not in self.punkt else 0
                q_tf[i][j] = question_unique_tokens[q_token] / question_len if q_token not in self.punkt else 0
        c_features = np.stack([c_em, c_tf], axis=-1)
        q_features = np.stack([q_em, q_tf], axis=-1)
        return c_features, q_features


@register('squad_dummy_features')
class SquadDummyFeatures(Component):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, batch):
        return [None] * len(batch)


@register('squad_ner_tags_preprocessor')
class SquadNerTagsPreprocessor(Component):
    """
    O -> O
    B-LOC -> LOC
    """
    def __init__(self, tokens_limit, *args, **kwargs):
        self.tokens_limit = tokens_limit
        self.PAD = '<PAD>'

    def __call__(self, ner_tags_batch, **kwargs):
        processed_batch = []
        for ner_tags in ner_tags_batch:
            ner_tags = [ner_tag.split('-')[-1] for ner_tag in ner_tags[:self.tokens_limit]]
            ner_tags = ner_tags + (self.tokens_limit - len(ner_tags)) * [self.PAD]
            processed_batch.append(ner_tags)
        return processed_batch


@register('squad_ner_tags_extractor')
class SquadNerTagsExtractor(Component):
    def __init__(self, ner_config, *args, **kwargs):
        config = json.load(open(expand_path(ner_config)))
        # remove tokenizer from ner pipeline
        config['chainer']['in'] = ['x_tokens']
        config['chainer']['pipe'] = config['chainer']['pipe'][1:]

        self.ner = build_model_from_config(config)

    def __call__(self, tokens_batch, **kwargs):
        batch_tags = [tags for tokens, tags in self.ner(tokens_batch)]
        return batch_tags