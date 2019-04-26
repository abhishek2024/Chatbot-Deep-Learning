# originally based on https://github.com/allenai/bilm-tf/blob/master/bilm/training.py

# Modifications copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import pprint
from deeppavlov.models.transformer_chit_chat.hacks_utils import spec_utters, yandex_api, stop_words
import numpy as np
import random
import re
import copy
import json
from logging import getLogger
from typing import List

# new

from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.nn_model import Serializable
from deeppavlov.models.transformer_chit_chat.model.transformer_model import TransformerModel
from deeppavlov.models.transformer_chit_chat.model.text import BertBPEVocab
from deeppavlov.models.transformer_chit_chat.model.utils import pad_sequence
from deeppavlov.models.tokenizers.utils import detokenize
import torch
import pathlib
log = getLogger(__name__)


punct = re.compile('[!.,?]')


def hello_bye(input_utter):
    # correct
    answers = None
    if [None for mtch in spec_utters.match_inttros if mtch in input_utter]:
        answers = random.sample(spec_utters.sample_inttros, 1)[0]
    if [None for mtch in spec_utters.match_ends if mtch in input_utter]:
        answers = random.sample(spec_utters.sample_ends, 1)[0]
    return answers


def start(input_utter):
    # correct
    if '/start' in input_utter:
        return 'Хочу пообщаться'


def split_text(text):
    text = text.split('<t>')
    return [t.strip() for t in text]


def clean_text(text):
    text = text.strip().lower()
    char_text = punct.sub('', text)
    char_text = ' '.join([spec_utters.MORPH.parse(word)[0].normal_form for word in char_text.split()])
    return split_text(text), split_text(char_text)


def clean_hypt(hypt):
    hypt = hypt.strip().lower()
    char_hypt = punct.sub('', hypt)
    char_hypt = ' '.join([spec_utters.MORPH.parse(word)[0].normal_form for word in char_hypt.split()])
    return char_hypt


def clean_pers(pers):
    text = ' <t> '.join(pers)
    return clean_text(text)


def clean_his(his):
    text = ' <t> '.join(his)
    text = yandex_api.send(text)
    return clean_text(text)


def len_filter(hypt, min_len=2):
    return len(hypt.split()) > min_len


def rm_sw_utter(utter, min_len=2):
    utter = punct.sub('', utter)
    words = utter.strip().lower().split()
    words = collections.OrderedDict(zip(words, [0]*len(words)))
    [words.pop(i, None)for i in stop_words.stop_words]
    utter = ' '.join(words.keys())
    return utter


def his_repeat_filter(cntx, hypt, his_len=4, max_repeat=2):
    his = cntx.his[1::2][-his_len:]
    his = set([rm_sw_utter(utter) for utter in his])
    res = his & set([rm_sw_utter(hypt)])
    return not bool(res)


def hypt_repeat_filter(cntx, hypt):
    if not hasattr(cntx, 'hypt_set'):
        cntx.hypt_set = set()
    hypt = punct.sub('', hypt)
    hypt = hypt.lower().strip()
    if hypt in cntx.hypt_set:
        return False
    else:
        cntx.hypt_set.add(hypt)
        return True



def cntx_analysis(cntx):
    cntx.persona_word_set = set(' '.join(cntx.char_persona).split())
    cntx.persona_word_set = cntx.persona_word_set - (cntx.persona_word_set & stop_words.stop_words)
    cntx.his_word_set = set(' '.join(cntx.char_his).split())
    cntx.his_word_set = cntx.his_word_set - (cntx.his_word_set & stop_words.stop_words)
    cntx.last_uttr_word_set = set(cntx.char_his[-1].split())
    cntx.last_uttr_word_set = cntx.last_uttr_word_set - (cntx.last_uttr_word_set & stop_words.stop_words)


def renew_hypt_conf(hypts, change_conf):
    hypts = copy.deepcopy(hypts)
    hypts = [(change_conf(conf, hypt), hypt) for conf, hypt in hypts]
    return hypts


def switch_hypt(cntx):
    hypts = cntx.hypts
    hypts = [(conf, hypt) for conf, hypt in hypts if len_filter(hypt, min_len=2)]
    # print(f'cntx.his[1::2] = {cntx.his[1::2]}')
    hypts = [(conf, hypt) for conf, hypt in hypts if his_repeat_filter(cntx, hypt, his_len=60, max_repeat=1)]
    hypts = [(conf, hypt) for conf, hypt in hypts if hypt_repeat_filter(cntx, hypt)]

    res_hypts = hypts

    def drop_conf_of_question(conf, hypt):
        if len(cntx.his) < 20 and '?' in hypt:
            #if last utter is question

            # if '?' in cntx.his[-1]:
            #     return 0
            bot_utters = cntx.his[1::2]
            decrease_pow = len([None for utter in bot_utters[-2:] if '?' in utter])
            # decrease_pow = len([None for utter in bot_utters[-2:] if '?' in utter])*2 - 1
            if decrease_pow < 1:
                return conf
            else:
                return 0
                # return conf/(2**decrease_pow)
                
        else:
            return conf

    res_hypts = renew_hypt_conf(res_hypts, drop_conf_of_question)

    res_hypts.sort(key=lambda x: x[0], reverse=True)
    res_hypts = res_hypts[: 3]
    if res_hypts:
        confs, answers = list(zip(*res_hypts))

        confs = np.array(confs)
        confs = confs/confs.sum()
        return np.random.choice(answers, p=confs)
    else:
        return random.sample(["Не знаю, что сказать... Как дела?",
                              "Как все сложно. Извини, не понимаю.",
                              "Не понимаю."], 1)[0]


def hacking(persona, his, hyp_answers, confs):
    def cntx(x): return x
    cntx.hypts = [(conf, ans) for conf, ans in zip(confs, hyp_answers)]
    cntx.persona, cntx.char_persona = clean_pers(persona)
    cntx.his, cntx.char_his = clean_his(his)
    hb_utter = hello_bye(cntx.char_his[-1])
    if hb_utter:
        return hb_utter
    st_utter = start(cntx.char_his[-1])
    if st_utter:
        return st_utter
    cntx_analysis(cntx)
    return switch_hypt(cntx)


@register('transformer_chit_chat')
class TransformerChitChat(Serializable):
    """
    """

    def __init__(self,
                 n_layers: int = 12,  # model config
                 n_pos_embeddings: int = 512,
                 embeddings_size: int = 768,
                 n_heads: int = 12,
                 dropout: float = 0.1,
                 embed_dropout: float = 0.1,
                 attn_dropout: float = 0.1,
                 ff_dropout: float = 0.1,
                 max_seq_len: int = 128,
                 sep_id_enable: bool = True,
                 beam_size: int = 10,  # 6
                 diversity_coef: float = 0,
                 diversity_groups: int = 10,  # 1,  # 6
                 annealing_topk: int = None,
                 annealing: float = 0,  # 0.5
                 length_penalty: float = 0.6,
                 n_segments: bool = True,
                 bert_mode: bool = True,
                 type_vocab_size: int = 4,
                 tie_weights: bool = True,
                 sample: bool = True,

                 bert_vocab_path: str = './vocab',  # vocab config
                 device: str = 'cuda',
                 #  device: str = 'cuda',
                 **kwargs) -> None:
        super().__init__(save_path='', **kwargs)

        self.model_config = lambda x: x  # not very nice but faster
        self.model_config.n_layers = n_layers
        self.model_config.n_pos_embeddings = n_pos_embeddings
        self.model_config.embeddings_size = embeddings_size
        self.model_config.n_heads = n_heads
        self.model_config.dropout = dropout
        self.model_config.embed_dropout = embed_dropout
        self.model_config.attn_dropout = attn_dropout
        self.model_config.ff_dropout = ff_dropout
        self.model_config.max_seq_len = max_seq_len
        self.model_config.sep_id_enable = sep_id_enable
        self.model_config.beam_size = beam_size
        self.model_config.diversity_coef = diversity_coef
        self.model_config.diversity_groups = diversity_groups
        self.model_config.annealing_topk = annealing_topk
        self.model_config.annealing = annealing
        self.model_config.length_penalty = length_penalty
        self.model_config.n_segments = n_segments
        self.model_config.bert_mode = bert_mode
        self.model_config.type_vocab_size = type_vocab_size
        self.model_config.tie_weights = tie_weights
        self.model_config.sample = sample

        self.device = device
        self.bert_vocab_path = str(pathlib.Path(bert_vocab_path).expanduser())
        self.load()

    @overrides
    def load(self) -> None:
        """Load model parameters from self.load_path"""
        self.vocab = BertBPEVocab.from_files(self.bert_vocab_path)
        self.transformer = TransformerModel(n_layers=self.model_config.n_layers,
                                            n_embeddings=len(self.vocab),
                                            n_pos_embeddings=self.model_config.n_pos_embeddings,
                                            embeddings_size=self.model_config.embeddings_size,
                                            padding_idx=self.vocab.pad_id,
                                            n_heads=self.model_config.n_heads,
                                            dropout=self.model_config.dropout,
                                            embed_dropout=self.model_config.embed_dropout,
                                            attn_dropout=self.model_config.attn_dropout,
                                            ff_dropout=self.model_config.ff_dropout,
                                            bos_id=self.vocab.bos_id,
                                            eos_id=self.vocab.eos_id,
                                            max_seq_len=self.model_config.max_seq_len,
                                            beam_size=self.model_config.beam_size,
                                            sample=self.model_config.sample,
                                            length_penalty=self.model_config.length_penalty,
                                            n_segments=self.model_config.n_segments,
                                            annealing_topk=self.model_config.annealing_topk,
                                            annealing=self.model_config.annealing,
                                            diversity_coef=self.model_config.diversity_coef,
                                            diversity_groups=self.model_config.diversity_groups,
                                            bert_mode=self.model_config.bert_mode,
                                            type_vocab_size=self.model_config.type_vocab_size,
                                            tie_weights=self.model_config.tie_weights,
                                            info_bos_id=self.vocab.info_bos_id,
                                            talker1_bos_id=self.vocab.talker1_bos_id,
                                            talker2_bos_id=self.vocab.talker2_bos_id,
                                            bos_token_id=self.vocab.bos_id,
                                            sep_token_id=self.vocab.sep_id,
                                            )
        self.transformer = self.transformer.to(self.device)
        path = self.load_path
        log.info(f'[loading from {path}]')
        state_dict = torch.load(path,
                                map_location=self.device)
        self.transformer.load_state_dict(state_dict['model'], strict=False)

    @overrides
    def save(self) -> None:
        """Save model parameters to self.save_path"""
        pass

    def context2tagged_context(self, utterances) -> List[float]:
        utters_ids = [self.vocab.string2ids(s) for s in utterances]
        tagged_utters_ids = []
        for i, ids in enumerate(utters_ids, 1):
            if i % 2 == 1:
                ids = [self.vocab.talker1_bos_id] + ids + [self.vocab.talker1_eos_id]
            else:
                ids = [self.vocab.talker2_bos_id] + ids + [self.vocab.talker2_eos_id]
            if tagged_utters_ids:
                tagged_utters_ids.append(self.vocab.sep_id)
            tagged_utters_ids.extend(ids)
        tagged_utters_ids = tagged_utters_ids[-128:]
        return tagged_utters_ids

    def drop_rich_msg(self, objs) -> List[str]:
        new_obj = []
        for obj in objs:
            if type(obj).__name__ == 'RichMessage':
                new_obj.append(obj.json()[0].get('content', ''))
            else:
                new_obj.append(obj)
        return new_obj

    def drop_max_len(self, objs, max_len=15) -> List[str]:
        new_obj = []
        for obj in objs:
            if type(obj).__name__ == 'RichMessage':
                new_obj.append(obj.json()[0].get('content', ''))
            else:
                new_obj.append(obj)
        return objs[-max_len:]

    persona = [
        "У меня есть парень.",
        "Я люблю суши.",
        "Я студентка.",
        "Я подрабатываю.",
        "Я хожу в бассейн.",

    def __call__(self, utterances_batch, history_batch, states_batch, *args, **kwargs) -> List[float]:
        history_batch = copy.deepcopy(history_batch)

        history_batch = [self.drop_rich_msg(his) for his in history_batch]
        history_batch = [self.drop_max_len(his,6) for his in history_batch]

        [history.append(utter) for utter, history in zip(utterances_batch, history_batch)]
        tagged_persona_batch = [self.context2tagged_context(self.persona) for _ in history_batch]
        tagged_persona_batch = [torch.tensor(d, dtype=torch.long, device=self.device) for d in tagged_persona_batch]
        tagged_persona_batch = pad_sequence(tagged_persona_batch,
                                            batch_first=True,
                                            padding_value=self.transformer.padding_idx)
        tagged_context_batch = [self.context2tagged_context(context) for context in history_batch]
        tagged_context_batch = [torch.tensor(d, dtype=torch.long, device=self.device) for d in tagged_context_batch]
        tagged_context_batch = pad_sequence(tagged_context_batch,
                                            batch_first=True,
                                            padding_value=self.transformer.padding_idx)
        predictions, confidence = self.transformer.predict([tagged_persona_batch, tagged_context_batch])
        predictions_batch = []
        for beam_pred, beam_conf in zip(predictions, confidence):
            lines = []
            for prediction, conf in zip(beam_pred, beam_conf):
                tokens = [id for id in prediction if id not in self.vocab.special_tokens_ids]
                sent = detokenize(self.vocab.ids2string(tokens).split())
                line = sent
                lines.append(line)
            predictions_batch.append(lines)
        predictions_batch = [hacking(self.persona, his, hyp_answers, confs) for his, hyp_answers, confs in zip(history_batch, predictions_batch, confidence)]
        return predictions_batch, confidence
