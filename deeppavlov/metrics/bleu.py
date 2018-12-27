# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
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

import re
from typing import List, Tuple, Any

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction, brevity_penalty, closest_ref_length

from deeppavlov.core.common.metrics_registry import register_metric
from deeppavlov.metrics.google_bleu import compute_bleu

SMOOTH = SmoothingFunction()


@register_metric('bleu_advanced')
def bleu_advanced(y_true: List[Any], y_predicted: List[Any],
                  weights: Tuple=(1,), smoothing_function=SMOOTH.method1,
                  auto_reweigh=False, penalty=True) -> float:
    """Calculate BLEU score

    Parameters:
        y_true: list of reference tokens
        y_predicted: list of query tokens
        weights: n-gram weights
        smoothing_function: SmoothingFunction
        auto_reweigh: Option to re-normalize the weights uniformly
        penalty: either enable brevity penalty or not

    Return:
        BLEU score
    """

    bleu_measure = sentence_bleu([y_true], y_predicted, weights, smoothing_function, auto_reweigh)

    hyp_len = len(y_predicted)
    hyp_lengths = hyp_len
    ref_lengths = closest_ref_length([y_true], hyp_len)

    bpenalty = brevity_penalty(ref_lengths, hyp_lengths)

    if penalty is True or bpenalty == 0:
        return bleu_measure

    return bleu_measure/bpenalty


@register_metric('bleu')
def bleu(y_true, y_predicted, tokenizer=lambda x: x.split()):
    """Calculate BLEU for strings or tokens"""
    if isinstance(y_true[0], str):
        y_true = (tokenizer(y_t.lower()) for y_t in y_true)
    if isinstance(y_predicted[0], str):
        y_predicted = (tokenizer(y_p.lower()) for y_p in y_predicted)
    return corpus_bleu([[y_t] for y_t in y_true], list(y_predicted))


@register_metric('google_bleu')
def google_bleu(y_true, y_predicted, tokenizer=lambda x: x.split()):
    """Calculate Google BLEU for strings or tokens"""
    if isinstance(y_true[0], str):
        y_true = (tokenizer(y_t.lower()) for y_t in y_true)
    if isinstance(y_predicted[0], str):
        y_predicted = (tokenizer(y_p.lower()) for y_p in y_predicted)
    return compute_bleu(([y_t] for y_t in y_true), y_predicted)[0]


def tokenize_13a(line):
    """
    Tokenizes an input line using a relatively minimal tokenization that is however equivalent to mteval-v13a, used by WMT.
    :param line: a segment to tokenize
    :return: the tokenized line
    """

    norm = line

    # language-independent part:
    norm = norm.replace('<skipped>', '')
    norm = norm.replace('-\n', '')
    norm = norm.replace('\n', ' ')
    norm = norm.replace('&quot;', '"')
    norm = norm.replace('&amp;', '&')
    norm = norm.replace('&lt;', '<')
    norm = norm.replace('&gt;', '>')

    # language-dependent part (assuming Western languages):
    norm = " {} ".format(norm)
    norm = re.sub(r'([\{-\~\[-\` -\&\(-\+\:-\@\/])', ' \\1 ', norm)
    norm = re.sub(r'([^0-9])([\.,])', '\\1 \\2 ', norm)  # tokenize period and comma unless preceded by a digit
    norm = re.sub(r'([\.,])([^0-9])', ' \\1 \\2', norm)  # tokenize period and comma unless followed by a digit
    norm = re.sub(r'([0-9])(-)', '\\1 \\2 ', norm)  # tokenize dash when preceded by a digit
    norm = re.sub(r'\s+', ' ', norm)  # one space only between words
    norm = re.sub(r'^\s+', '', norm)  # no leading space
    norm = re.sub(r'\s+$', '', norm)  # no trailing space

    return norm.split()
