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

from typing import List, Generator, Any

from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords as nltk_stopwords
import pymorphy2

from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register
from deeppavlov.models.tokenizers.utils import detokenize, ngramize, replace
from deeppavlov.core.common.log import get_logger

logger = get_logger(__name__)


@register('ru_tokenizer')
class RussianTokenizer(Component):
    """
    Tokenize or lemmatize a list of documents.
    Return list of tokens or lemmas, without sentencizing.
    Works only for Russian language.
    """

    def __init__(self, stopwords: list = None, ngram_range: List[int] = None, lemmas=False,
                 lowercase: bool = None, alphas_only: bool = None,
                 replace: dict = None, **kwargs):
        """
        :param stopwords: a set of words to skip
        :param ngram_range: range for producing ngrams, ex. for unigrams + bigrams should be set to
        [1, 2], for bigrams only should be set to [2, 2]
        :param lemmas: weather to perform lemmatizing or not while tokenizing, currently works only
        for the English language
        :param lowercase: perform lowercasing or not
        :param alphas_only: should filter numeric and alpha-numeric types or not
        :param replace: a dict with String types to replace and corresponding replacers.
        Ex.: {'isnumeric': 'NUM', 'isalpha': 'WORD'}
        """

        if replace is None:
            replace = {}

        if ngram_range is None:
            ngram_range = [1, 1]

        if stopwords == 'nltk' or 'NLTK_STOPWORDS':
            stopwords = set(nltk_stopwords.words('russian'))

        self.stopwords = stopwords or []
        self.tokenizer = ToktokTokenizer()
        self.lemmatizer = pymorphy2.MorphAnalyzer()
        self.ngram_range = tuple(ngram_range)  # cast JSON array to tuple
        self.lemmas = lemmas
        self.lowercase = lowercase
        self.alphas_only = alphas_only
        self.tok2morph = {}

        cast_replace = {}

        for item_type, replacer in replace:
            if item_type == 'isnumeric':
                cast_replace[str.isnumeric] = replacer
            elif item_type == 'isalpha':
                cast_replace[str.isalpha] = replacer
            elif item_type == 'isdigit':
                cast_replace[str.isdigit] = replacer
            elif item_type == 'isalnum':
                cast_replace[str.isalnum] = replacer
            elif item_type == 'isdecimal':
                cast_replace[str.isdecimal] = replacer
            elif item_type == 'isupper':
                cast_replace[str.isupper] = replacer

        self.replace = cast_replace

    def __call__(self, batch):
        if isinstance(batch[0], str):
            if self.lemmas:
                return list(self._lemmatize(batch))
            else:
                return list(self._tokenize(batch))
        if isinstance(batch[0], list):
            return [detokenize(doc) for doc in batch]
        raise TypeError(
            "StreamSpacyTokenizer.__call__() is not implemented for `{}`".format(type(batch[0])))

    def _tokenize(self, data: List[str], lowercase=True) -> Generator[List[str], Any, None]:
        """
        Tokenize a list of documents.
        :param data: a list of documents to process
        :param batch_size: the number of documents to process at once
        :param lowercase: whether to perform lowercasing or not
        :return: a single processed doc generator
        """
        # DEBUG
        # size = len(data)
        if self.lowercase is None:
            _lowercase = lowercase
        else:
            _lowercase = self.lowercase

        for i, doc in enumerate(data):
            # DEBUG
            # logger.info("Tokenize doc {} from {}".format(i, size))
            tokens = self.tokenizer.tokenize(doc)
            if _lowercase:
                tokens = [t.lower() for t in tokens]
            processed_doc = self._pipe(tokens)
            yield processed_doc

    def _lemmatize(self, data: List[str]) -> Generator[List[str], Any, None]:
        """
        Lemmatize a list of documents.
        :param data: a list of documents to process
        :return: a single processed doc generator
        """
        # DEBUG
        # size = len(data)
        tokenized_data = list(self._tokenize(data))

        for i, doc in enumerate(tokenized_data):
            # DEBUG
            # logger.info("Lemmatize doc {} from {}".format(i, size))
            lemmas = []
            for token in doc:
                try:
                    lemma = self.tok2morph[token]
                except KeyError:
                    lemma = self.lemmatizer.parse(token)[0].normal_form
                    self.tok2morph[token] = lemma
                lemmas.append(lemma)
            processed_doc = self._pipe(lemmas)
            yield processed_doc

    def _filter(self, items, alphas_only=True):
        """
        Make ngrams from a list of tokens/lemmas
        :param items: list of tokens, lemmas or other strings to form ngrams
        :param alphas_only: should filter numeric and alpha-numeric types or not
        :return: filtered list of tokens/lemmas
        """
        if self.alphas_only is None:
            _alphas_only = alphas_only
        else:
            _alphas_only = self.alphas_only

        if _alphas_only:
            filter_fn = lambda x: x.isalpha() and not x.isspace() and x not in self.stopwords
        else:
            filter_fn = lambda x: not x.isspace() and x not in self.stopwords

        return list(filter(filter_fn, items))

    def set_stopwords(self, stopwords):
        self.stopwords = stopwords

    def _pipe(self, items, ngram_range=(1, 1)):
        """
        :param items: tokens or lemmas to replace, filter and ngramize
        :param ngram_range: range for producing ngrams, ex. for unigrams + bigrams should be set to
        (1, 2), for bigrams only should be set to (2, 2)
        :return: processed tokens/lemmas
        """
        _ngram_range = self.ngram_range or ngram_range
        # replaced = replace(items, self.replace)
        filtered = self._filter(items)
        processed_doc = ngramize(filtered, ngram_range=_ngram_range)
        return processed_doc


