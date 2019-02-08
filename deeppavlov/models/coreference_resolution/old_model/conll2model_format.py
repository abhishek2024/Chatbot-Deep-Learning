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

import collections
from typing import Dict, Union


class DocumentState(object):
    """
    Class that verifies the conll document and creates on its basis a new dictionary.
    """

    def __init__(self) -> None:
        self.doc_key = None
        self.text = []
        self.text_speakers = []
        self.speakers = []
        self.sentences = []
        self.clusters = collections.defaultdict(list)
        self.stacks = collections.defaultdict(list)

    def assert_empty(self) -> None:
        assert self.doc_key is None, 'self.doc_key is None'
        assert len(self.text) == 0, 'len(self.text) == 0'
        assert len(self.text_speakers) == 0, 'len(self.text_speakers)'
        assert len(self.sentences) == 0, 'len(self.sentences) == 0'
        assert len(self.speakers) == 0, 'len(self.speakers) == 0'
        assert len(self.clusters) == 0, 'len(self.clusters) == 0'
        assert len(self.stacks) == 0, 'len(self.stacks) == 0'

    def assert_finalizable(self) -> None:
        assert self.doc_key is not None, 'self.doc_key is not None finalizable'
        assert len(self.text) == 0, 'len(self.text) == 0_finalizable'
        assert len(self.text_speakers) == 0, 'len(self.text_speakers) == 0_finalizable'
        assert len(self.sentences) > 0, 'len(self.sentences) > 0_finalizable'
        assert len(self.speakers) > 0, 'len(self.speakers) > 0_finalizable'
        assert all(
            len(s) == 0 for s in self.stacks.values()), 'all(len(s) == 0 for s in self.stacks.values())_finalizable'

    def finalize(self) -> Dict:
        merged_clusters = []
        for c1 in self.clusters.values():
            existing = None
            for m in c1:
                for c2 in merged_clusters:
                    if m in c2:
                        existing = c2
                        break
                if existing is not None:
                    break
            if existing is not None:
                # print("Merging clusters (shouldn't happen very often.)")
                existing.update(c1)
            else:
                merged_clusters.append(set(c1))
        merged_clusters = [list(c) for c in merged_clusters]

        return {"doc_key": self.doc_key,
                "sentences": self.sentences,
                "speakers": self.speakers,
                "clusters": merged_clusters}


def normalize_word(word: str) -> str:
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word


def handle_line(line: str, document_state: DocumentState) -> Union[None, Dict]:
    """
    Updates the state of the document_state class in a read string of conll document.

    Args:
        line: line from conll document
        document_state: analyzing class

    Returns: document_state, or Nothing

    """
    if line.startswith("#begin"):
        document_state.assert_empty()
        # row = line.split()
        #    document_state.doc_key = 'bc'+'{0}_{1}'.format(row[2][1:-2],row[-1])
        return None
    elif line.startswith("#end document"):
        document_state.assert_finalizable()
        return document_state.finalize()
    else:
        row = line.split('\t')
        if len(row) == 1:
            document_state.sentences.append(tuple(document_state.text))
            del document_state.text[:]
            document_state.speakers.append(tuple(document_state.text_speakers))
            del document_state.text_speakers[:]
            return None
        assert len(row) >= 12, 'len < 12'

        word = normalize_word(row[3])
        coref = row[-1]
        if document_state.doc_key is None:
            document_state.doc_key = 'bc' + '{0}_{1}'.format(row[0], row[1])

        speaker = row[8]

        word_index = len(document_state.text) + sum(len(s) for s in document_state.sentences)
        document_state.text.append(word)
        document_state.text_speakers.append(speaker)

        if coref == "-":
            return None

        for segment in coref.split("|"):
            if segment[0] == "(":
                if segment[-1] == ")":
                    cluster_id = segment[1:-1]  # here was int
                    document_state.clusters[cluster_id].append((word_index, word_index))
                else:
                    cluster_id = segment[1:]  # here was int
                    document_state.stacks[cluster_id].append(word_index)
            else:
                cluster_id = segment[:-1]  # here was int
                start = document_state.stacks[cluster_id].pop()
                document_state.clusters[cluster_id].append((start, word_index))
        return None


def conll2modeldata(conll_str: str) -> Dict:
    """
    Converts the document into a dictionary, with the required format for the model.

    Args:
        conll_str: dict with conll string

    Returns:
        dict like:
            {
              "clusters": [[[1024,1024],[1024,1025]],[[876,876], [767,765], [541,544]]],
              "doc_key": "nw",
              "sentences": [["This", "is", "the", "first", "sentence", "."], ["This", "is", "the", "second", "."]],
              "speakers": [["spk1", "spk1", "spk1", "spk1", "spk1", "spk1"], ["spk2", "spk2", "spk2", "spk2", "spk2"]]
            }
    """
    model_file = None
    document_state = DocumentState()
    line_list = conll_str.split('\n')
    for line in line_list:
        document = handle_line(line, document_state)
        if document is not None:
            model_file = document
    return model_file
