from collections import defaultdict
import ujson as json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from deeppavlov.core.models.estimator import Estimator
from deeppavlov.core.common.registry import register

@register("unimorph_vectorizer")
class UnimorphDictionaryVectorizer(Estimator):

    def __init__(self, save_path, load_path, **kwargs):
        load_path = Path(load_path).with_suffix(".json")
        save_path = Path(save_path).with_suffix(".json")
        super().__init__(save_path=save_path, load_path=load_path, **kwargs)
        if self.load_path.exists():
            self.load()

    @property
    def pos_number(self):
        return len(self.pos_)

    @property
    def dim(self):
        return len(self.pos_) + len(self.pos_features_)

    def load(self):
        # load_path = Path(self.load_path).with_suffix(".json")
        with open(self.load_path, "r", encoding="utf8") as fin:
            data = json.load(fin)
        self.pos_, self.pos_features_ = data["pos"], data["features"]
        self.pos_codes_ = {tag: i for i, tag in enumerate(self.pos_ + self.pos_features_)}
        self.word_indexes_ = defaultdict(list)
        for word, elem in data["word_indexes"].items():
            self.word_indexes_[word] = [list(map(int, x.split(","))) for x in elem.split(";")]
        return

    def save(self):
        to_save = {"pos": self.pos_, "features": self.pos_features_, "word_indexes": {}}
        for word, word_indexes in self.word_indexes_.items():
            to_save["word_indexes"][word] = ";".join(",".join(map(str, elem)) for elem in word_indexes)
        save_path = Path(self.save_path).with_suffix(".json")
        with open(save_path, "w", encoding="utf8") as fout:
            json.dump(to_save, fout, ensure_ascii=False)

    def tag_to_features(self, tag):
        tag = tag.split(";")
        pos = tag[0]
        answer = [pos]
        for x in tag[1:]:
            answer.append("{}_{}".format(pos, x))
        return answer

    def fit(self, data: List[Tuple[str, str, str]], *args, **kwargs):
        """

        Args:
            data: a list of triples of the form (lemma, word, UniMorph tag)

        Returns:

        """
        pos_tags = set()
        pos_features = set()
        for lemma, word, tag in data:
            features = self.tag_to_features(tag)
            pos_tags.add(features[0])
            pos_features.update(features[1:])
        self.pos_ = sorted(pos_tags)
        self.pos_features_ = sorted(pos_features)
        self.pos_codes_ = {tag: i for i, tag in enumerate(self.pos_ + self.pos_features_)}
        # second pass to transform tag string to vectors
        self.word_indexes_ = defaultdict(list)
        for lemma, word, tag in data:
            features = self.tag_to_features(tag)
            indexes = [self.pos_codes_[feat] for feat in features]
            self.word_indexes_[word].append(indexes)
        return

    def _get_word_vector(self, word):
        answer = np.zeros(shape=(self.dim,), dtype=float)
        vectors = self.word_indexes_[word]
        if len(vectors) > 0:
            for vector in vectors:
                answer[vector] += 1.0
            answer /= len(vectors)
        return answer

    def __call__(self, data: List) -> np.ndarray:
        """
        Transforms words to one-hot encoding according to the dictionary.

        Args:
            data: the batch of words

        Returns:
            a 3D array. answer[i][j][k] = 1 iff data[i][j] is the k-th word in the dictionary.
        """
        # if isinstance(data[0], str):
        #     data = [[x for x in re.split("(\w+|[,.])", elem) if x.strip() != ""] for elem in data]
        max_length = max(len(x) for x in data)
        answer = np.zeros(shape=(len(data), max_length, self.dim), dtype=int)
        for i, sent in enumerate(data):
            for j, word in enumerate(sent):
                answer[i, j] = self._get_word_vector(word)
        return answer

